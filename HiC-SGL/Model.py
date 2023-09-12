import torch_geometric.nn as gnn 
import torch.nn as nn
import torch.nn.functional as F
import torch as th 
import torch 

class FFN(torch.nn.Module):
    def __init__(self, embed_size, ff_hidden_size = None):
        super(FFN, self).__init__()
        if not ff_hidden_size:
            ff_hidden_size = embed_size
        self.fc1 = torch.nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = torch.nn.Linear(ff_hidden_size, embed_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MHA(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MHA, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = torch.nn.Linear(embed_size, self.head_dim * heads)
        self.keys = torch.nn.Linear(embed_size, self.head_dim * heads)
        self.queries = torch.nn.Linear(embed_size, self.head_dim * heads)
        
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.heads, self.head_dim)
        return x.permute(2, 0, 1, 3)
        
    def forward(self, x, bias, mask): # x(b, n, d), bias, mask(b, n, n)
        batch_size = x.shape[0]
        
        values = self.split_heads(self.values(x), batch_size)
        keys = self.split_heads(self.keys(x), batch_size)
        queries = self.split_heads(self.queries(x), batch_size)
        #q(h, b, n, d/h), k(h, b, n, d/h), v(h, b, n, d/h)
        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2))#(h, b, n, n)
        attention_scores = attention_scores + bias
        attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))
        #attn(b, h, n, n)
        attention_probs = F.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        #ATN.append(attention_probs)
        output = torch.matmul(attention_probs, values)#attn(h, b, n, d/h)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.heads * self.head_dim)
        return self.fc_out(output)

class GT(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.MHA = MHA(dim, head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)

    def forward(self, x, bias, mask):
        x_normalized = self.norm1(x)
        attention_output = self.MHA(x_normalized, bias, mask)
        x = x + attention_output
        
        x_normalized = self.norm2(x)
        feed_forward_output = self.ffn(x_normalized)
        x = x + feed_forward_output
        
        return x
    
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
        #BIAS.append(bias)
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

