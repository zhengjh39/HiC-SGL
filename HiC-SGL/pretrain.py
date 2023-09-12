import argparse
import os 

parser = argparse.ArgumentParser(description='str')
parser.add_argument('--cuda', type = str, default = '0')
parser.add_argument('--dir', type = str, default = 'Ramani/')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

from time import time 
import os 
from utils import *
import Model 
from DataGenerator import *
import torch.nn as nn
from tqdm import trange, tqdm
if __name__ == '__main__':
    data_dir = args.dir
    print('GCL Pretrain on dataset', data_dir, 'CUDA', args.cuda)
    import json 
    with open(os.path.join(data_dir, 'config.JSON')) as f:
        config = json.load(f)

    #max_node = config['max_node']
    batch_size = config['batch_size']
    weight_path = os.path.join(data_dir, 'GCLweight')
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    for c in range(len(config['chrom_list'])):
        print('PreTrain on chrom', config['chrom_list'][c])
        cells, cell_feat = get_cells(data_dir, c)

        neibor_limit = get_neibor_limit(cells)
        test_edge, test_label, train_pos = edgeData(cells)

        train_set = GraphSet(cells, neibor_limit, train_pos, pretrain = True)
        train_loader = DataLoader(train_set, config['batch_size'], shuffle = True, num_workers = config['num_workers'] , drop_last = True )

        negset = NegSet(cells, 1)
        negloader = DataLoader(negset, 32, shuffle = False, num_workers = 16, collate_fn = lambda d: d)

        model = get_model(data_dir, c, 'init', 'cuda')
        loss_func = nn.BCELoss()
        opt = th.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

        epoch_num = config['pretrain_epoch']
        start_epoch = 0 
        device = 'cuda'
        weight_path_c = os.path.join(weight_path, 'chr'+str(c))
        if not os.path.exists(weight_path_c):
            os.mkdir(weight_path_c)

        for epoch in range(start_epoch, start_epoch + epoch_num):
            model.train()
            train_neg = []
            for edge in negloader:
                edge = th.cat(edge)
                train_neg.append(edge)
            train_neg = th.cat(train_neg)
            train_edge = th.cat([train_pos, train_neg])
            train_label = th.cat([th.ones(train_pos.shape[0]), th.zeros(train_neg.shape[0])])
            train_set.getedge(train_edge, train_label)
            pretrain_epoch(model, train_loader, opt,  device, epoch, False)

        th.save(model.sub_encoder.state_dict(), os.path.join(weight_path_c, 'w'))
