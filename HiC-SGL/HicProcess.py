import torch as th
from torch_geometric.data import Data
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os 
import pandas as pd
import math
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import trange, tqdm
import matplotlib.pyplot as plt 
def generate_chrom_start_end(config):
    # fetch info from config
	genome_reference_path = config['genome_reference_path']
	chrom_list = config['chrom_list']
	res = config['resolution']
	data_dir = config['data_dir']
	
	#print ("generating start/end dict for chromosome")
	chrom_size = pd.read_table(genome_reference_path, sep="\t", header=None)
	chrom_size.columns = ['chrom', 'size']
	# build a list that stores the start and end of each chromosome (unit of the number of bins)
	chrom_start_end = np.zeros((len(chrom_list), 2), dtype='int')
	for i, chrom in enumerate(chrom_list):
		size = chrom_size[chrom_size['chrom'] == chrom]
		size = size['size'][size.index[0]]
		n_bin = int(math.ceil(size / res))
		chrom_start_end[i, 1] = chrom_start_end[i, 0] + n_bin
		if i + 1 < len(chrom_list):
			chrom_start_end[i + 1, 0] = chrom_start_end[i, 1]
	
	# print("chrom_start_end", chrom_start_end)
	np.save(os.path.join(data_dir, "chrom_start_end.npy"), chrom_start_end)

def data2triplets(config, data, chrom_start_end):
    	# fetch info from config
	res = config['resolution']
	chrom_list = config['chrom_list']

	pos1 = np.array(data['pos1'])
	pos2 = np.array(data['pos2'])
	bin1 = np.floor(pos1 / res).astype('int')
	bin2 = np.floor(pos2 / res).astype('int')
	
	chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
	cell_id = np.array(data['cell_id'].values).astype('int')
	count = np.array(data['count'].values)
	
	del data
	
	new_chrom1 = np.ones_like(bin1, dtype='int') * -1
	new_chrom2 = np.ones_like(bin1, dtype='int') * -1
	for i, chrom in enumerate(tqdm(chrom_list)):
		mask = (chrom1 == chrom)
		new_chrom1[mask] = i
		# Make the bin id for chromosome 2 starts at the end of the chromosome 1, etc...
		bin1[mask] += chrom_start_end[i, 0]
		mask = (chrom2 == chrom)
		new_chrom2[mask] = i
		bin2[mask] += chrom_start_end[i, 0]
	
	# Exclude the chromosomes that showed in the data but are not selected.
	data = np.stack([cell_id, new_chrom1, new_chrom2, bin1, bin2], axis=-1)
	mask = (data[:, 1] >= 0) & (data[:, 2] >= 0)
	count = count[mask]
	data = data[mask]
	bin1, bin2 = data[:, 3], data[:, 4]
	new_bin1 = np.minimum(bin1, bin2)
	new_bin2 = np.maximum(bin1, bin2)
	data[:, 3] = new_bin1
	data[:, 4] = new_bin2
    
	unique, inv, unique_counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
	new_count = np.zeros_like(unique_counts, dtype='float32')
	for i, iv in enumerate(inv):
		new_count[iv] += count[i]
	
	return unique, new_count 
	

def process_raw(config):
	data_dir = config['data_dir']

	generate_chrom_start_end(config)
	chrom_start_end = np.load(os.path.join(data_dir, "chrom_start_end.npy"))
	data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
        
	unique, new_count = data2triplets(config, data, chrom_start_end)


	intra_contacts = unique[:, 1] == unique[:, 2]
	inter_contacts = unique[:, 1] != unique[:, 2]
	
	intra_data = unique[intra_contacts]
	intra_count = new_count[intra_contacts]
	intra_data = intra_data[:, [0, 1, 3, 4]]

	np.save(os.path.join(data_dir, "data.npy"), intra_data, allow_pickle=True)
	np.save(os.path.join(data_dir, "weight.npy"), intra_count.astype('float32'), allow_pickle=True)
	
	np.save(os.path.join(data_dir, "inter_data.npy"), unique[inter_contacts], allow_pickle=True)
	np.save(os.path.join(data_dir, "inter_weight.npy"), new_count[inter_contacts].astype('float32'), allow_pickle=True)
    
def negsample_score(score, size):
    n = score.shape[0]
    s = score.view(-1).numpy()
    size = min(size, s[s > 0].shape[0])
    if size == 0:
         return np.array([[0], [0]])
    t = np.random.choice(np.array(range(s.shape[0])), size, replace = False, p = s/s.sum())
    i = t // n
    j = t % n 
    neg = np.stack([i, j])
    return th.from_numpy(neg)

def prepare_data(data_dir, dense_thre = [-1, -1], c = 0, neg_num = 5, drop_ratio = 0.1, config = None):
    data = np.load(os.path.join(data_dir, 'data.npy'))
    chr_start_end = np.load(os.path.join(data_dir, 'chrom_start_end.npy'))
    weight = np.load(os.path.join(data_dir, 'weight.npy'))
    label_path = os.path.join(data_dir, 'label_info.pickle')
    label = np.zeros(max(data[:, 0]) + 1)

    if os.path.exists(label_path):
         label = pd.read_pickle(label_path)['cell type']
    start = chr_start_end[c][0]
    end = chr_start_end[c][1]
    bin_num = end - start
    mask = (data[:, 2] != data[:, 3])
    x = th.from_numpy(data[mask])
    weight = th.from_numpy(weight[mask])     
    t, sp = th.unique(x[:, 0], return_counts = True)
    sp = sp.tolist()
    weight = th.split(weight, sp)
    cells = th.split(x, sp)

    if config and 'cell type' in config:
        weight = [weight[i] for i in range(len(cells)) if label[cells[i][0][0]] in config['cell type']]
        cells = [cells[i] for i in range(len(cells)) if label[cells[i][0][0]] in config['cell type']]
        label = [l for l in label if l in config['cell type']]
 
    pair_num = th.tensor([w.sum() for w in weight])
    dtmax = dense_thre[1]  if dense_thre[1] != -1 else 1e10
    #print(np.percentile(pair_num.numpy(), 50))
    cidx = th.argwhere( (pair_num > dense_thre[0]) & (pair_num < dtmax) ).view(-1)
    #cidx = th.argwhere(pair_num > dense_thre[0]).view(-1)

    cells = [cells[i] for i in cidx]
    weight = [weight[i] for i in cidx]
    label = [label[i] for i in cidx]
    pair_num = [pair_num[i] for i in cidx]
    #print(np.percentile(pair_num, 50))
    for i in range(len(cells)):
        mask = cells[i][:, 1] == c 
        cells[i] = cells[i][mask]
        weight[i] = weight[i][mask]

    class cellset(Dataset):
        def __init__(self, x, w, l):
            super(Dataset, self).__init__()
            self.x = x
            self.w = w
            self.l = l
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            x = self.x[idx]
            w = self.w[idx]
            l = self.l[idx]
            
            tmp = Data()
            if x.shape == 0:
                tmp.edge_index = th.tensor([[], []])
            else:
                tmp.edge_index = th.stack((x[:,2] - start, x[:,3] - start))
                
            tmp.num_nodes = bin_num
            tmp.vidx = th.tensor(range(start, end))
            tmp.cidx = th.tensor([idx, c])
            tmp.edge_attr = w
            tmp.label = l
            tmp.emask = th.rand(tmp.edge_index.shape[1])
            tmp.drop_ratio = drop_ratio
            test_pos = tmp.edge_index[:, tmp.emask >= (1 - drop_ratio)]
            adj = coo_matrix((w, (tmp.edge_index[0], tmp.edge_index[1])), shape = (bin_num, bin_num))
            adj = adj.toarray()
            score = th.ones((bin_num, bin_num))
            score = th.triu(score, 1)
            score[adj > 0] = 0

            test_neg = negsample_score(score, test_pos.shape[1] * neg_num)
            tmp.test_neg = test_neg
            
            return tmp
        
    dataset =  cellset(cells, weight, label)
    loader = DataLoader(dataset, 32, shuffle=False, num_workers= 64, collate_fn = lambda d: d)
    #print(next(iter(loader)))
    cells = []
    for batch in loader:
        for g in batch:
             cells.append(g.clone())
    
    graphpath = os.path.join(data_dir, 'cellgraph')
    if not os.path.exists(os.path.join(data_dir, 'cellgraph')):
        os.mkdir(graphpath)
    th.save(cells, os.path.join(graphpath, str(c)))
    return cells

def gen_cellatr(cells, cell_dim, dir, pool = None):
    c = cells[0].cidx[1].item()
    atrpath = os.path.join(dir, 'cellatr')
    if not os.path.exists(atrpath):
        os.mkdir(atrpath)
    
    x = []
    for g in cells:
        mask = g.emask < 1 - g.drop_ratio
        edge_attr = g.edge_attr[mask]
        edge_index = g.edge_index[:, mask]
        adj = coo_matrix((edge_attr, (edge_index[0], edge_index[1])), shape = (g.num_nodes, g.num_nodes)).toarray()
        if pool:
            adj = th.from_numpy(adj).unsqueeze(dim = 0)
            adj = pool(adj).squeeze().numpy()

        x.append(adj.reshape(-1))
    x = np.stack(x)
    cell_feat = TruncatedSVD(n_components = cell_dim).fit_transform(x)
    cell_feat = th.from_numpy(cell_feat).float()
    th.save(cell_feat, os.path.join(atrpath, str(c)))
    return cell_feat

import argparse
import json 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='str')
    parser.add_argument('--dir', type = str)
    args = parser.parse_args()
    data_dir = args.dir 
    with open(os.path.join(data_dir, 'config.JSON')) as f:
        config = json.load(f)
    print('Process raw data...')
    process_raw(config)
    print('Translate triplet table to Graph and generate cell feature...')
    for c in trange(len(config['chrom_list'])):
        cells = prepare_data(data_dir, config['dense_thre'], c, 5, config['drop_ratio'], config)
        gen_cellatr(cells, config['cell_dim'], data_dir, False)
    
    
