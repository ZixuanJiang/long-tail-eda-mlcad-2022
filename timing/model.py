import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            fcs.append(torch.nn.LayerNorm(sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                #fcs.append(torch.nn.Dropout(p=dropout))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class Classifier(torch.nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)-1):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            fcs.append(torch.nn.LayerNorm(sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                fcs.append(torch.nn.Dropout(p=0.2))
        fcs.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class AllConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, h1, h2, out_nf, mlp_h1=64, mlp_h2=64, mlp_h3=64):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg = MLP(in_nf * 2 + in_ef, mlp_h1, mlp_h2, mlp_h3, 1 + h1 + h2)
        self.MLP_reduce = MLP(in_nf + h1 + h2, mlp_h1, mlp_h2, mlp_h3, out_nf)

    def edge_udf(self, edges):
        x = self.MLP_msg(torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1))
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'ef1': f1 * k, 'ef2': f2 * k}

    def forward(self, g, nf, ef):
        with g.local_scope():
            g.ndata['nf'] = nf
            g.edata['ef'] = ef
            g.apply_edges(self.edge_udf)
            g.update_all(fn.copy_e('ef1', 'ef1'), fn.sum('ef1', 'nf1'))
            g.update_all(fn.copy_e('ef2', 'ef2'), fn.max('ef2', 'nf2'))
            x = torch.cat([g.ndata['nf'], g.ndata['nf1'], g.ndata['nf2']], dim=1)
            x = self.MLP_reduce(x)
            return x

class TimingGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_node_outputs):
        super().__init__()
        self.conv1 = AllConv(num_node_features, num_edge_features, 64, 64, 64)
        self.conv2 = AllConv(64, num_edge_features, 64, 64, 64)
        self.conv3 = AllConv(64, num_edge_features, 64, 64, 64)
        self.output = Classifier(64, 256, 1024, num_node_outputs)

    def forward(self, g):
        x = self.conv1(g, g.ndata['node_features'], g.edata['edge_features'])
        x = self.conv2(g, x, g.edata['edge_features'])
        x = self.conv3(g, x, g.edata['edge_features'])
        x = self.output(x)
        return x

class AllConvBatch(torch.nn.Module):
    def __init__(self, in_nf, in_ef, h1, h2, out_nf, mlp_h1=64, mlp_h2=64, mlp_h3=64):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg = MLP(in_nf * 2 + in_ef, mlp_h1, mlp_h2, mlp_h3, 1 + h1 + h2)
        self.MLP_reduce = MLP(in_nf + h1 + h2, mlp_h1, mlp_h2, mlp_h3, out_nf)

    def edge_udf(self, edges):
        x = self.MLP_msg(torch.cat([edges.src['nf'], edges.dst['nf'], edges.data['ef']], dim=1))
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'ef1': f1 * k, 'ef2': f2 * k}

    def forward(self, block, nf):
        with block.local_scope():
            nf_src = nf
            nf_dst = nf[:block.number_of_dst_nodes()]
            block.srcdata['nf'] = nf
            block.dstdata['nf'] = nf_dst
            block.edata['ef'] = block.edata['edge_features']
            #print(block.srcdata['nf'].shape)
            #print(block.edata['ef'].shape)
            block.apply_edges(self.edge_udf)
            block.update_all(fn.copy_e('ef1', 'ef1'), fn.sum('ef1', 'nf1'))
            block.update_all(fn.copy_e('ef2', 'ef2'), fn.max('ef2', 'nf2'))
            x = torch.cat([block.dstdata['nf'], block.dstdata['nf1'], block.dstdata['nf2']], dim=1)
            x = self.MLP_reduce(x)
            return x

class TimingGCNBatch(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_node_outputs):
        super().__init__()
        self.conv1 = AllConvBatch(num_node_features, num_edge_features, 64, 64, 64)
        self.conv2 = AllConvBatch(64, num_edge_features, 64, 64, 64)
        self.conv3 = AllConvBatch(64, num_edge_features, 64, 64, 64)
        self.output = Classifier(64, 256, 1024, num_node_outputs)

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        x = self.conv3(blocks[2], x)
        x = self.output(x)
        return x
   
