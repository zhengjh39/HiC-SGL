from DataGenerator import *
from sklearn.metrics import average_precision_score,roc_auc_score
from torch_geometric.utils import degree
from GCL.models import DualBranchContrast
import GCL.losses as L
from tqdm import tqdm
import os 
import json 
import Model
from sklearn.decomposition import PCA
def get_neibor_limit(cells):
    d = []
    node_nums = {}
    for g in cells:
        node_nums[g.num_nodes] = 1
    
    for g in cells:
        d1 = degree(g.edge_index[0], g.num_nodes)
        d2 = degree(g.edge_index[1], g.num_nodes)
        d.append(d1 + d2)
    d = th.cat(d).numpy()
    d1 = int(np.percentile(d, 95))
    d2 = int(np.percentile(d, 80))
    neibor_limit = [d1 * 2]
    if (d1 + d1 * d2 + 1) * 2 < np.percentile(list(node_nums.keys()), 50)/10:
        neibor_limit.append(d1 * d2 * 2)
    return neibor_limit

def evaluate_cellwise(model, config, cells, device):
    test_edge, test_label, train_pos = edgeData(cells)
    neibor_limit = get_neibor_limit(cells)
    test_set = GraphSet(cells, neibor_limit, test_edge, test_label)
    test_loader = DataLoader(test_set, config['batch_size'], shuffle = False, num_workers = config['num_workers'], drop_last = False )
    
    model.eval()
    label = []
    score = []
    with th.no_grad():
        for data, y in test_loader:
            for k, v  in data.items():  
                data[k] = v.to(device)
            y = y.to(device)
            _y = model(data)
            label.append(y.cpu())
            score.append(_y.cpu())
           
    label = th.cat(label)
    score = th.cat(score)

    aupr = np.ones(len(cells))
    auc = np.ones(len(cells))
    idx, count = th.unique(test_edge[:, 0], return_counts = True)
    label = th.split(label, count.tolist())
    score = th.split(score, count.tolist())
    for i in range(len(score)):
        m1 = average_precision_score(label[i], score[i])
        aupr[idx[i]] = m1
        m2 = roc_auc_score(label[i], score[i])
        auc[idx[i]] = m2
    print(aupr.mean(), auc.mean())
    return auc, aupr

def evaluate(model, valid_loader, loss_func, device):
    model.eval()
    bces = []
    aps = []
    aucs = []
    with th.no_grad():
        for data, y in valid_loader:
            for k, v  in data.items():  
                data[k] = v.to(device)
            y = y.to(device)
            _y = model(data)
            loss = loss_func(_y, y)
            bces.append(loss.item())
            aps.append(average_precision_score(y.cpu(), _y.detach().cpu()))
            aucs.append(roc_auc_score(y.cpu(), _y.detach().cpu()))
        bce = round(sum(bces)/len(bces), 4)
        ap = round(sum(aps)/len(aps), 4)
        auc = round(sum(aucs)/len(aucs), 4)
    return bce, ap, auc
    
def train_epoch(model, opt, loss_func, train_loader, device, test_loader, epoch, val_interval = 1, show = False):
        bces = []
        aps = []
        def idty(x):
            return x
        f = tqdm if show else idty
        for data, y in f(train_loader):
            for k, v in data.items():  
                data[k] = v.to(device)

            y = y.to(device)
            _y = model(data)
            loss = loss_func(_y, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bces.append(loss.item())
            aps.append(average_precision_score(y.cpu(), _y.detach().cpu()))

        train_bce = round(sum(bces)/len(bces), 4)
        train_ap = round(sum(aps)/len(aps), 4)
        if (epoch + 1) % val_interval == 0:
            val_bce, val_ap, val_auc = evaluate(model, test_loader, loss_func, device)
            print(epoch, 'train_bce:', train_bce, 'train_ap:',train_ap,
                'val_bce', val_bce, 'val_ap', val_ap, 'val_auc', val_auc)

def pretrain_epoch(model, train_loader, opt,  device, epoch, show = True):
    model.train()
    losses = []
    def idty(x):
        return x
    f = tqdm if show else idty
    for aug1, aug2 in f(train_loader):
        for k in aug1.keys():  
            aug1[k] = aug1[k].to(device)
            aug2[k] = aug2[k].to(device)
            
        g1 = model(aug1, pretrain = True)
        g2 = model(aug2, pretrain = True)
        contrast_model = DualBranchContrast(loss = L.InfoNCE(tau=0.2), mode='G2G').to(device)
        loss = contrast_model(g1 = g1 , g2 = g2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
    loss = round(sum(losses)/len(losses), 4)
    if epoch % 5 == 0:
        print('epoch:',epoch, 'InfoNCE:', loss)

def impute_cell(raw_cell, model):
    count = []
    edges = []
    neibor_limit = get_neibor_limit(raw_cell)
    test_edge, test_label, train_pos = edgeData(raw_cell)

    train_set = GraphSet(raw_cell, neibor_limit, train_pos)
    for g in raw_cell:
        n = g.num_nodes
        cell_id = th.tensor([g.cidx[0]]).repeat(n * n)
        t = th.tensor(range(n))
        u = t.repeat_interleave(n)
        v = t.repeat(n)
        edge = th.stack([cell_id, u, v], dim = 1)
        edge = edge[u < v]
        count.append(edge.shape[0])
        edges.append(edge)

    edges = th.cat(edges)
    label = th.ones(edges.shape[0])
    train_set.getedge(edges, label)
    loader = DataLoader(train_set, 1024, shuffle = False, num_workers = 64, drop_last = False)
    pred = []
    with th.no_grad():
        model.eval()
        for data, y in tqdm(loader):
            for key, value  in data.items():  
                data[key] = value.to('cuda')
            pred.append(model(data))

    pred = th.cat(pred).cpu()
    pred = th.split(pred, count)
    impute_maps = []
    for i in range(len(raw_cell)):
        g = raw_cell[i]
        score = pred[i]
        n = g.num_nodes
        cell_map = th.zeros(n * n)
        cell_map[u < v] = score
        impute_maps.append(cell_map.view(n, n))
    #pred = th.cat(pred ).cpu()
    return impute_maps 

def get_cell_embed(data_dir):
    cell_embed = []
    chr_start_end = np.load(os.path.join(data_dir, 'chrom_start_end.npy'))
    bin_num = chr_start_end[-1][1]
    with open(os.path.join(data_dir, 'config.JSON')) as f:
        config = json.load(f)
    for c in range(len(config['chrom_list'])):
        cells = th.load(os.path.join(data_dir, 'cellgraph', str(c)))
        cell_feat = [th.load(os.path.join(data_dir, 'cellatr', str(c)))]
        cell_feat = th.stack(cell_feat, dim = 1)
        neibor_limit = get_neibor_limit(cells)
        dim = config['node_dim']
        c_dim = config['cell_dim'] 
        model = Model.LinkPredictor(dim, 6, 4, sum(neibor_limit) + 2, c_dim, cell_feat, bin_num)
        weight_path = os.path.join(data_dir, 'weight', 'chr'+ str(c))
        model.load_state_dict(th.load(os.path.join(weight_path, str(config['train_epoch'] - 1))))
        with th.no_grad():
            embed = model.get_embed()
        embed = embed.view(cell_feat.shape[0], -1)
        cell_embed.append(embed)
    cell_embed = th.cat(cell_embed, dim = -1)
    mu = cell_embed.mean(dim = 0)
    std = cell_embed.std(dim = 0)
    bn_embed = (cell_embed - mu)/std    
    bn_embed =  PCA(n_components = 64).fit_transform(bn_embed)
    return bn_embed

def get_cells(data_dir, c = 0):
    cells_c = th.load(os.path.join(data_dir, 'cellgraph', str(c)))
    for i in range(len(cells_c)):
        cells_c[i].cidx = th.tensor([i, 0])
    cells = cells_c

    cell_feat = [th.load(os.path.join(data_dir, 'cellatr', str(c)))]
    cell_feat = th.stack(cell_feat, dim = 1)
    return cells, cell_feat

def get_model(data_dir, c = 0, state = 'init', device = 'cpu'):
    cells_c = th.load(os.path.join(data_dir, 'cellgraph', str(c)))
    chr_start_end = np.load(os.path.join(data_dir, 'chrom_start_end.npy'))
    bin_num = chr_start_end[-1][1]
    with open(os.path.join(data_dir, 'config.JSON')) as f:
        config = json.load(f)
    neibor_limit = get_neibor_limit(cells_c)

    cell_feat = [th.load(os.path.join(data_dir, 'cellatr', str(c)))]
    cell_feat = th.stack(cell_feat, dim = 1).to(device)

    dim = config['node_dim']
    c_dim = config['cell_dim'] 
    model = Model.LinkPredictor(dim, 6, 4, sum(neibor_limit) + 2, c_dim, cell_feat, bin_num)
    if state == 'pretrained': 
        model.sub_encoder.load_state_dict(th.load(os.path.join(data_dir, 'GCLweight','chr' + str(c), 'w')))
    if state == 'trained':
        model.sub_encoder.load_state_dict(th.load(os.path.join(data_dir, 'GCLweight','chr' + str(c), str(config['train_epoch'] - 1))))
    if device != 'cpu':
        model.cuda(device)
    return model