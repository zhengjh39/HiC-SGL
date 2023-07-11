import torch_geometric.nn as gnn 
import torch.nn as nn
import torch.nn.functional as F
import torch as th 
class FFN(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h

class MHA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.Wq = nn.Linear(in_dim, hidden_dim, bias = False)
        self.Wk = nn.Linear(in_dim, hidden_dim, bias = False)
        self.Wv = nn.Linear(in_dim, out_dim, bias = False)

    def forward(self, H, bias, mask):
        Q, K, V = self.Wq(H), self.Wk(H), self.Wv(H)
        d = Q.shape[-1]
        atn = Q @ K.transpose(1,2) / d ** (1/2)
        
        atn = atn + bias 
        atn[mask == 0] = -(1e10)
        atn = th.softmax(atn ,dim = -1)
        return atn @ V

class GT(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.MHAs = nn.ModuleList([MHA(dim, dim//head, dim//head) for i in range(head)])
        self.h = head 
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.fc = nn.Linear(dim, dim, bias = False)

    def forward(self, H,  bias, mask):
        H =  self.ln1(H)
        Hs = []
        for mha in self.MHAs:
            Hs.append(mha(H, bias, mask))
        H = H + self.fc(th.cat(Hs, dim = -1))
        H = self.ffn(self.ln2(H)) + H

        return H
    
class V_feat(nn.Module):
    def __init__(self, v_dim, node_num, bin_num):
        super().__init__()   
        self.vidx = nn.Embedding(bin_num, v_dim)
        max_pos = node_num * node_num // 2 + 50 
        self.pos = nn.Embedding(max_pos + 1, v_dim)
        self.deg = nn.Embedding(node_num, v_dim)
    
    def forward(self, Data):
        vidx = self.vidx(Data['vidx'])
        pos = self.pos(Data['pos'])
        deg = self.deg(Data['deg'])
        H = vidx + pos + deg 
        return H      
     
class E_feat(nn.Module):
    def __init__(self, node_num):
        super().__init__()
        self.node = node_num
        edge_num = node_num * node_num
        self.mlp = nn.Sequential(nn.LayerNorm(edge_num), nn.Linear(edge_num, edge_num), nn.LayerNorm(edge_num), nn.ReLU(),  
                                 nn.Linear(edge_num, edge_num), nn.LayerNorm(edge_num))

    def forward(self, Data):
        
        edge_num = self.node  * self.node 

        dist = Data['dist'].float().view(-1, edge_num)
        
        bias = th.tanh(self.mlp(dist)).view(-1, self.node, self.node)

        return bias 

class SubEncoder(nn.Module):
    def __init__(self, dim, layer, head, node_num, bin_num):
        super().__init__()
        self.v_feat = V_feat(dim, node_num, bin_num)
        self.e_feat = E_feat(node_num)
        self.GTs = nn.ModuleList([GT(dim, head) for i in range(layer)])
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, Data):
        H = self.v_feat(Data)
        bias = self.e_feat(Data)
        mask = Data['mask']
        for GT in self.GTs:
            H = GT(H, bias, mask)
        H = self.ln(H)
        
        return H

class CellEncoder(nn.Module):
    def __init__(self, dim, cell_feat):
        super().__init__()
        self.cell_feat = cell_feat
        cell_dim = cell_feat.shape[-1]
        self.mlp = nn.Sequential(nn.Linear(cell_dim, cell_dim), nn.LayerNorm(cell_dim), nn.ReLU(),  
                                 nn.Linear(cell_dim, dim), nn.LayerNorm(dim))
    
    def forward(self, cidx):
        cidx = cidx.long()
        cell_feat = self.cell_feat[[cidx[:, 0], cidx[:, 1]]]
        cell_embed = self.mlp(cell_feat)
   
        return cell_embed 

    
class LinkPredictor(nn.Module):
    def __init__(self, dim, layer, head, node_num, cdim, cell_feat, bin_num):
        super().__init__()
        self.sub_encoder = SubEncoder(dim, layer, head, node_num, bin_num)
        self.cell_encoder = CellEncoder(cdim, cell_feat)
        self.decoder = nn.Sequential(nn.Linear(cdim + dim * 2 , cdim + dim * 2), nn.ReLU(), 
                                    nn.Linear(cdim + dim * 2, 1), nn.Sigmoid())
        
        self.project = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.LayerNorm(dim * 2), 
                                     nn.ReLU(), nn.Linear(dim * 2, dim * 2))
    
    def get_embed(self, cell_feat = None):
        if cell_feat == None:
            cell_feat = self.cell_encoder.cell_feat
        return self.cell_encoder.mlp(cell_feat)
    
    def forward(self, Data, pretrain = False):
        local_embed = self.sub_encoder(Data)
        global_embed = self.cell_encoder(Data['cidx'])

        if pretrain:
            H = local_embed
            H = th.cat([H[:, 0], H[:, 1]], dim = 1)
            H = self.project(H)
            return H
        
        y =  self.decoder(th.cat([global_embed, local_embed[:, 0], local_embed[:, 1]], dim = -1) )
        return y.view(-1)