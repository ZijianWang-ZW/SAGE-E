import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGEELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 1)))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # aggregator function
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            
            # update function
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1)))
            return g.ndata['h']

        
# we adopt a 4 layer SAGE-E model here which owns best performance under this situation
class SAGEE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGEE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEELayer(ndim_in, edim, 50, activation))
        self.layers.append(SAGEELayer(50, edim, 50, activation))
        self.layers.append(SAGEELayer(50, edim, 25, activation))
        self.layers.append(SAGEELayer(25, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats
