import os
import os.path
import torch
import dgl
import math
import numpy as np
from dgl import load_graphs
from dgl.data.dgl_dataset import DGLDataset
import matplotlib.pyplot as plt
import seaborn as sns

class CirDelay(DGLDataset):
    def __init__(
            self, 
            name = 'CirDelay',
            raw_dir = '/home/local/eda10/jayliu/projects/timing/DELAY', 
            num_classes = 100,
            verbose = 0,
            ):

        self._train_name = ['usb_cdc_core', 'salsa20', 'wbqspiflash', 'cic_decimator', 'aes256', 'aes_cipher', 'genericfir', 'usb', 'jpeg_encoder', 'aes192', 'xtea', 'spm', 'BM64', 'blabla', 'picorv32a']
        self._test_name = ['zipdiv', 'usbf_device', 'aes128', 'synth_ram']
        self.num_classes = num_classes
        self.ver = verbose
        super(CirDelay, self).__init__(name, raw_dir=raw_dir)

    def has_cache(self):
        for graph in self._train_name: 
            if not os.path.exists("{}/{}.graph.bin".format(self._raw_dir, graph)):
                return False
        for graph in self._test_name: 
            if not os.path.exists("{}/{}.graph.bin".format(self._raw_dir, graph)):
                return False
        return True

    def process(self):
        self.load()

    def load(self):
        self._g_train = []
        self._g_test = []

        for graph in self._train_name:
            g = load_graphs("{}/{}.graph.bin".format(self._raw_dir, graph))[0][0]
            g.ndata['node_net_delays'] = torch.log(0.001+g.ndata['node_net_delays'].max(dim=1).values)
            self._g_train.append(g)
        self._g_train_batch = dgl.batch(self._g_train)

        for graph in self._test_name:
            g = load_graphs("{}/{}.graph.bin".format(self._raw_dir, graph))[0][0]
            g.ndata['node_net_delays'] = torch.log(0.001+g.ndata['node_net_delays'].max(dim=1).values)
            self._g_test.append(g)
        self._g_test_batch = dgl.batch(self._g_test)

        self.num_node_features = g.ndata['node_features'].shape[1]
        self.num_edge_features = g.edata['edge_features'].shape[1]

        self._y_min = min(self._g_train_batch.ndata['node_net_delays'].min(), self._g_test_batch.ndata['node_net_delays'].min())
        self._y_max = min(self._g_train_batch.ndata['node_net_delays'].max(), self._g_test_batch.ndata['node_net_delays'].max())
        if self.num_classes > 20:
            self._y_dist = (self._y_max - self._y_min)/self.num_classes  * 0.9
        else:
            self._y_dist = (self._y_max - self._y_min)/self.num_classes  
        self._g_train_batch.ndata['label'] = torch.min((self._g_train_batch.ndata['node_net_delays']-self._y_min)//self._y_dist, torch.zeros_like(self._g_train_batch.ndata['node_net_delays'])+(self.num_classes-1))
        self._g_test_batch.ndata['label'] = torch.min((self._g_test_batch.ndata['node_net_delays']-self._y_min)//self._y_dist, torch.zeros_like(self._g_test_batch.ndata['node_net_delays'])+(self.num_classes-1))
        for g in self._g_train:
            g.ndata['label'] = torch.min((g.ndata['node_net_delays']-self._y_min)//self._y_dist, torch.zeros_like(g.ndata['node_net_delays'])+(self.num_classes-1))
        for g in self._g_test:
            g.ndata['label'] = torch.min((g.ndata['node_net_delays']-self._y_min)//self._y_dist, torch.zeros_like(g.ndata['node_net_delays'])+(self.num_classes-1))

        self._g = dgl.batch([self._g_train_batch, self._g_test_batch])

        train_label_count = []
        test_label_count = []
        for i in range(self.num_classes):
            train_label_count.append(torch.count_nonzero(self._g_train_batch.ndata['label'] == i).item())
            test_label_count.append(torch.count_nonzero(self._g_test_batch.ndata['label'] == i).item())

        if self.ver:
            for i in range(self.num_classes):
                print("class {}: {:.5f}   {:.5f}".format(i, train_label_count[i]/sum(train_label_count)*100, test_label_count[i]/sum(test_label_count)*100))
            print(sum(train_label_count), sum(test_label_count))
            print(train_label_count)
            print(test_label_count)

        for i, name in enumerate(self._train_name):
            count = []
            for j in range(self.num_classes):
                count.append(torch.count_nonzero(self._g_train[i].ndata['label'] == j).item())
            if self.ver:
                print(name, sum(count), count)

        for i, name in enumerate(self._test_name):
            count = []
            for j in range(self.num_classes):
                count.append(torch.count_nonzero(self._g_test[i].ndata['label'] == j).item())
            if self.ver:
                print(name, sum(count), count)

    def __len__(self):
        return self._g_train_batch.num_nodes()

    def graph(self):
        return self._g

    def train_test_nids(self, split=0.8, seed=314):
        np.random.seed(seed)
        num_nodes = self._g.num_nodes()
        nids = np.arange(num_nodes)
        np.random.shuffle(nids)
        num_train = int(num_nodes * split)
        train_idx = nids[:num_train]
        test_idx = nids[num_train:]
        if self.ver:
            train_label_count = []
            test_label_count = []
            for i in range(self.num_classes):
                train_label_count.append(torch.count_nonzero(self._g.ndata['label'][train_idx] == i).item())
                test_label_count.append(torch.count_nonzero(self._g.ndata['label'][test_idx] == i).item())
                print("class {}: {:.5f}   {:.5f}".format(i, train_label_count[i]/num_train*100, test_label_count[i]/(num_nodes-num_train)*100))

            print('Imbalance Ratio:', max(train_label_count) / min(train_label_count))
            div = 0
            gini = 0
            train_label_count.sort()
            for i in range(self.num_classes):
                pi = train_label_count[i] / num_train
                div += pi * math.log10(self.num_classes * pi)
                gini += train_label_count[i] * (2 * i - self.num_classes + 1)
            print('Imbalance Divergence:', div)
            print('Gini Coefficient:', gini / (self.num_classes * num_train))
            print(len(train_label_count[:50]))
            print(sum(train_label_count[:50]) / num_train)
            print(len(train_label_count[-10:]))
            print(sum(train_label_count[-10:]) / num_train)

        return train_idx, test_idx

    def train_graph(self):
        return self._g_train_batch

    def train_graphs(self):
        return self._g_train

    def test_graph(self):
        return self._g_test_batch
    
    def test_graphs(self):
        return self._g_test

if __name__ == '__main__':
    dataset = CirDelay(verbose=1)
    dataset.train_test_nids()
