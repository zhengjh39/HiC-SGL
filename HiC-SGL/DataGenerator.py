from torch_geometric.transforms import ToUndirected
import torch as th
import torch_geometric 
from torch_geometric.data import Data
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import random 
from torch_geometric.utils import subgraph
import networkx 
import torch.nn.functional as F
from scipy.sparse import coo_matrix 
import matplotlib.pyplot as plt 
def negsample_score(score, size):
    n = score.shape[0]
    s = score.reshape(-1)
    size = min(size, s[s > 0].shape[0])
    t = np.random.choice(np.array(range(s.shape[0])), size, replace = False, p = s/s.sum())
    i = t // n
    j = t % n 
    neg = np.stack([i, j])
    return th.from_numpy(neg)

def getDeg(g):
    return torch_geometric.utils.degree(g.edge_index[0],g.num_nodes).int()

def getPos(g, max_node):
    if g.num_nodes < 2:
        print(g.vidx, g.edge_index)
    u , v = 0, 1
    g1 = torch_geometric.utils.to_networkx(g)
    g1.remove_node(v)
    p1 = networkx.shortest_path_length(g1, source=u)
    p1[v] = 1
    d1 = np.ones(g.num_nodes) * max_node
    d1[list(p1.keys())] = list(p1.values())
    
    g2 = torch_geometric.utils.to_networkx(g)
    g2.remove_node(u)
    p2 = networkx.shortest_path_length(g2 ,source = v)
    p2[u] = 1

    d2 = np.ones(g.num_nodes) * max_node
    d2[list(p2.keys())] = list(p2.values())
    
    d = d1 + d2
    pos = np.ones(g.num_nodes)
    for i in range(g.num_nodes):
        pos[i] = 1 + min(d1[i], d2[i]) + (d[i].item()//2) * (d[i].item()//2 + d[i] % 2 +1)
    max_pos = max_node * max_node // 2 + 50
    pos[pos > max_pos] = max_pos
    return th.from_numpy(pos).int()

def getDis(g):
    A = coo_matrix((g.edge_attr, (g.edge_index[0], g.edge_index[1])), shape = (g.num_nodes, g.num_nodes))
    A = th.from_numpy(A.toarray())
    A = A + th.eye(g.num_nodes) * -1
    return A

def dropout_node(g, p = 0.1):

    edge_index = g.edge_index.clone() 
    num_nodes = g.num_nodes
    prob = th.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    node_mask[0] = True
    node_mask[1] = True
    edge_index, edge_attr = subgraph(node_mask, edge_index, g.edge_attr,
                            num_nodes = num_nodes, relabel_nodes = True)
    subg = Data(vidx = g.vidx[node_mask], edge_index =  edge_index,
                edge_attr = edge_attr, cidx = g.cidx, num_nodes = g.vidx[node_mask].shape[0])
    #eidx, edge_attr, mask = torch_geometric.utils.remove_isolated_nodes(edge_index, edge_attr, num_nodes = subg.num_nodes)
    #subg = Data(vidx = subg.vidx[mask], edge_index = eidx, edge_attr = edge_attr,
    #            cidx = g.cidx, num_nodes = subg.vidx[mask].shape[0])
    return subg

def dropout_edge(g, p = 0.1):
    if g.edge_index.shape[1] == 0:
        return g.clone()
    edge_index = g.edge_index
    prob = th.rand(g.edge_index.shape[1])
    edge_mask = prob > p 
    edge_mask[0] =  True 

    row, col = edge_index 
    edge_mask[row > col] = False 
    edge_index = edge_index[:, edge_mask]
    edge_index = th.cat([edge_index, edge_index.flip(0)], dim =1)
    edge_attr = g.edge_attr[edge_mask]
    edge_attr = th.cat([edge_attr, edge_attr])
    
    subg = Data(vidx = g.vidx, edge_index = edge_index, edge_attr = edge_attr, 
                cidx = g.cidx, num_nodes = g.num_nodes)
    #eidx, edge_attr, mask = torch_geometric.utils.remove_isolated_nodes(edge_index, edge_attr, num_nodes = g.num_nodes)
    #subg = Data(vidx = g.vidx[mask], edge_index = eidx, edge_attr = edge_attr,
    #            cidx = g.cidx, num_nodes = g.vidx[mask].shape[0])
    
    return subg

def mask_feature(g, p = 0.1):
    g = g.clone()
    prob = th.rand(g.num_nodes)
    feat_mask = prob < p
    vidx = g.vidx
    rand_idx = th.randint(0, len(vidx), (vidx[feat_mask].shape[0], ))
    vidx[feat_mask] = vidx[rand_idx]
    return Data(vidx = vidx, edge_index = g.edge_index, cidx = g.cidx, 
                edge_attr = g.edge_attr, num_nodes = g.num_nodes)

def shuffle_node(g):
    g = g.clone()
    rp = list(range(2, g.num_nodes))
    random.shuffle(rp)
    rp = th.tensor([0, 1] + rp).long()
    g.edge_index.apply_(lambda d: rp[d])
    vidx = g.vidx[rp]

    return Data(vidx = vidx, edge_index = g.edge_index, edge_attr = g.edge_attr,
                cidx = g.cidx, num_nodes = g.num_nodes)

def trans(g, max_node):
    data = {}
    data['vidx'] = F.pad(g.vidx, (0, max_node - g.num_nodes))
    data['pos'] = F.pad(getPos(g, max_node) , (0, max_node - g.num_nodes))
    data['deg'] = F.pad(getDeg(g), (0, max_node - g.num_nodes))
    data['vmask'] = F.pad(th.ones(g.num_nodes, ), (0, max_node - g.num_nodes))
    data['mask'] = F.pad(th.ones((g.num_nodes, g.num_nodes)), (0, max_node - g.num_nodes, 0, max_node - g.num_nodes))
    data['dist'] = F.pad(getDis(g), (0, max_node - g.num_nodes, 0, max_node - g.num_nodes))
    data['cidx'] = g.cidx
    return data

def Aug(g):
    aug_func = [dropout_edge, dropout_node, mask_feature, shuffle_node]
    c = list(range(len(aug_func)))
    random.shuffle(c)
    g1 = aug_func[c[0]](g)
    g2 = aug_func[c[1]](g1)
    return g2

def draw(g):
    label = g.vidx.tolist()
    label = {i: label[i] for i in range(len(label))}
    g = torch_geometric.utils.to_networkx(g)
    networkx.draw(g, with_labels = True, labels = label)
    plt.show()

import random
def random_choice(iterable, n):
    reservoir = list(iterable)
    random.shuffle(reservoir)
    return reservoir[: n]

from time import time  
class GraphSet(Dataset):
    def __init__(self, graph, nlimit, edge, label = None, pretrain = False):
        self.graph = graph
        self.edge = edge.long() 
        self.label = label
        self.nlimit = nlimit 
        self.pretrain = pretrain
    
    def __len__(self):
        return self.edge.shape[0]
    
    def getedge(self, edge, label):
        self.edge = edge.long()
        self.label = label 

    def k_hop_subgraph(self, idx,  nlimit = [10, 20]):
        i, u, v = self.edge[idx]
        g = self.graph[i]
        node_idx = th.tensor([u, v])
        train_mask = g.emask < 0.9 
        edge_index = g.edge_index[:, train_mask]
        edge_attr = g.edge_attr[train_mask]
        
        row, col = edge_index
        edge_index = edge_index[:, (col - row) > 1]
        edge_attr = edge_attr[(col - row) > 1]
        next_link = [list(range(g.num_nodes - 1)), list(range(1, g.num_nodes))]
        next_link = th.tensor(next_link)
        edge_index = th.cat([edge_index, next_link], dim = 1)
        edge_attr = th.cat([edge_attr, th.ones(next_link.shape[1])])

        num_nodes = g.num_nodes
        m = (edge_index[0] != u ) | (edge_index[1] != v)
        edge_index = edge_index[:, m]
        edge_attr = edge_attr[m]

        edge_index = th.cat([edge_index, edge_index.flip(0)], dim = 1)
        edge_attr = th.cat([edge_attr, edge_attr])
        row, col = edge_index
        node_mask = row.new_empty(num_nodes, dtype=th.bool)
        edge_mask = row.new_empty(row.size(0), dtype=th.bool)

        subsets = node_idx
        add_set = subsets

        for i in range(len(nlimit)):
            node_mask.fill_(False)
            node_mask[add_set] = True
            th.index_select(node_mask, 0, row, out = edge_mask)
            add_set = col[edge_mask]
            add_set = set(add_set.tolist()) - set(subsets.tolist())
            add_set = random_choice(add_set, nlimit[i])
            add_set = th.tensor(list(add_set))
            #print(th.unique(add_set))
            subsets = th.cat([subsets, add_set])
        
        sub = subsets
        node_mask.fill_(False)
        node_mask[sub] = True 
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]
        rp = th.ones(num_nodes).long()
        rp[sub] = th.tensor(range(sub.shape[0]))
        edge_index = rp[edge_index]  
        subg = Data(vidx = g.vidx[sub], edge_index = edge_index, edge_attr = edge_attr, 
                    cidx = g.cidx, num_nodes = len(sub))
        return subg 
    
    def __getitem__(self, idx):
        subg = self.k_hop_subgraph(idx, self.nlimit)
        maxnode = sum(self.nlimit) + 2
        if self.pretrain :
            aug1 = Aug(subg)
            aug2 = Aug(subg)
            return trans(aug1, maxnode), trans(aug2, maxnode)

        return trans(subg, maxnode), self.label[idx]

from HicProcess import negsample
class NegSet(Dataset):
    def __init__(self, graph, neg_num):
        self.graph = graph
        self.neg_num = neg_num
    def __len__(self):
        return len(self.graph)
    
    def __getitem__(self, idx):
        g = self.graph[idx]
        n = g.num_nodes
        mask = g.emask
        train_mask = mask < 1 - g.drop_ratio
        train_pos = g.edge_index[:, train_mask]
        u, v = train_pos

        ex_edge = th.stack([u, u + 1], dim = 0)
        #edge_index = th.cat([g.A_edge_index, g.test_neg, ex_edge], dim = 1)
        edge_index = th.cat([g.edge_index, g.test_neg, ex_edge], dim = 1)
        
        train_neg = negsample(n, self.neg_num * train_pos.shape[1], edge_index)
        u = train_neg[0]
        v = train_neg[1] 
        cell_id = th.tensor([idx]).repeat(u.shape[0])
        triplet = th.stack([cell_id, u, v], dim = 1)
        return triplet


def edgeData(cells):
    train_pos = []
    testedge = []
    testlabel = []
    for i in range(len(cells)):
        g = cells[i]
        train_mask = (g.emask < 1 - g.drop_ratio) & (g.emask >= 0)
        u = g.edge_index[:, train_mask][0]
        v = g.edge_index[:, train_mask][1]
        cell_id = th.tensor([i]).repeat(u.shape[0])
        triplet = th.stack([cell_id, u, v], dim = 1)
            #mask = u < v - 1
        train_pos.append(triplet)

        test_mask = g.emask >= 0.9
        test_pos = g.edge_index[:, test_mask]
        test_neg = g.test_neg 
        test_edge = th.cat([test_pos, test_neg], dim = 1)
        test_label = th.cat([th.ones(test_pos.shape[1]), th.zeros(test_neg.shape[1])])
        u = test_edge[0]
        v = test_edge[1]
        cell_id = th.tensor([i]).repeat(u.shape[0])
        triplet = th.stack([cell_id, u, v], dim = 1)
        testedge.append(triplet)
        testlabel.append(test_label)

    train_pos = th.cat(train_pos)
    testedge = th.cat(testedge)
    testlabel = th.cat(testlabel)
    
    return testedge, testlabel, train_pos
