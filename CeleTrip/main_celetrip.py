import pickle
import random
import os
import sys
import logging

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

import dgl
import dgl.data
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn import GraphConv
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling, SumPooling


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.optim as optim


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import math
import networkx as nx
import time
from scipy.stats import t
from math import radians, cos, sin, asin, sqrt


random_seed = 42
def seed_everything(seed=random_seed):
    '''
    fix random seed
    :param random_seed:
    :return:
    '''
    random.seed(seed) # Python random module
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # Current CPU
    torch.cuda.manual_seed(seed) # Current GPU
    torch.backends.cudnn.benchmark = False # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
seed_everything(random_seed)

class MultiLocationDataset(DGLDataset):
    def __init__(self, maingraph_feature, subgraph_feature,
                 location_dict = None):
        
        self.maingraph_feature = maingraph_feature
        self.subgraph_feature = subgraph_feature

        super().__init__(name='MultiLocationDataset')
        
    def process(self):
        maingraph_feature = self.maingraph_feature["maingraph_info"]
        subgraph_feature = self.subgraph_feature["subgraphs_info"]
        print('The length of main graph : ',len(maingraph_feature))
        # init lists of item
        self.graphs, self.sub_graphs, self.special_nodes_lists = [], [], []
        self.labels, self.event_features, self.entity_features = [], [], []
        cnt = 0
        for main_id in maingraph_feature:
            main_info = maingraph_feature[main_id]
            sub_info = subgraph_feature[main_id]

            self.graphs.append(main_info[0])
            self.entity_features.append(main_info[1])
            self.event_features.append(main_info[2])

            self.sub_graphs.append(sub_info[0])
            self.special_nodes_lists.append(sub_info[1])
            self.labels.append(sub_info[2])
            cnt+=1

    def __getitem__(self,i):
        return self.graphs[i], self.sub_graphs[i], self.special_nodes_lists[i], self.labels[i], self.event_features[i], self.entity_features[i]
    
    def __len__(self):
        return len(self.graphs)
    

def get_stats(array, conf_interval=False, name=None, stdout=False, logout=False):
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n-1)
        err_bound = t_value * se
    else:
        err_bound = std

    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound


def get_batch_id(num_nodes:torch.Tensor):
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)
    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k



class OrientedPool(torch.nn.Module):
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(OrientedPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity
        self.cos_score = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.att_linear = nn.Linear(2, 1, bias=True)
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, need_guide = False, special_nodes_idx = None):
        score = self.score_layer(graph, feature).squeeze()
        
        if need_guide:

            graph.ndata["sim_score"] = torch.ones_like(score)
            graph.ndata["feat"] = feature
            graph_list = dgl.unbatch(graph)

            for each_graph, nodes_idx in zip(graph_list, special_nodes_idx):
                 for node_idx in nodes_idx:
                    if node_idx == -1: 
                         continue
                    node_feature = each_graph.ndata["feat"][node_idx]

                    node_feature = node_feature.repeat(each_graph.num_nodes(), 1)
                    cos_s = self.cos_score(node_feature, each_graph.ndata["feat"])
                    each_graph.ndata["sim_score"] = each_graph.ndata["sim_score"] * cos_s

            graph = dgl.batch(graph_list)
            sim_score = graph.ndata["sim_score"]
            score = torch.reshape(score, (score.shape[0], 1))
            sim_score = torch.reshape(sim_score, (sim_score.shape[0], 1))
            con_score = torch.cat([score, sim_score], 1)
            score = torch.flatten(torch.tanh(self.att_linear(con_score)))
            
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_modes='flatten'):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,
                                negative_slope=alpha, residual=residual)
        assert agg_modes in ['flatten', 'mean']
        self.agg_modes = agg_modes

    def forward(self, bg, feats):
        feats = self.gat_conv(bg, feats)
        if self.agg_modes == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)
        return feats
    
        
class Word_Doc_Block(nn.Module):
    """
    Version: Gloabl Pool without classifier
    @refer: https://github.com/dmlc/dgl/tree/0.9.x/examples/pytorch/sagpool
    """
    def __init__(self, config):
        super(Word_Doc_Block, self).__init__()
        self.dropout = config.feat_drops

        self.word_doc_gat_layer = nn.ModuleList()
        
        gat_in_dim = config.word_in_dim
        gat_out_dim = config.hid_dim
        for i in range(config.num_layers):
            self.word_doc_gat_layer.append(GATLayer(gat_in_dim, gat_out_dim,
                                                    config.num_heads, config.feat_drops,
                                                    config.attn_drops, config.alpha,
                                                    config.residuals, agg_modes=config.agg_modes))
            gat_in_dim = gat_out_dim * config.num_heads
        
        concat_dim = config.num_layers * (config.hid_dim * config.num_heads)
        self.pool = OrientedPool(concat_dim, ratio=config.pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()
        
        self.lin1 = torch.nn.Linear(concat_dim * 2, config.hid_dim)
        self.lin3 = torch.nn.Linear(config.hid_dim, config.out_dim)

    def forward(self, word_doc_graph, feat, special_nodes_idx, device='cuda'):
        one_graph = word_doc_graph.to(device)
        gat_res = []
        for gat_layer in self.word_doc_gat_layer:
            feat = gat_layer(one_graph, feat.float())
            feat = F.relu(feat) 
            gat_res.append(feat)
        
        gat_res = torch.cat(gat_res, dim=-1)
        one_graph, feat, _ = self.pool(one_graph, gat_res, need_guide = True, special_nodes_idx = special_nodes_idx)
        feat = torch.cat([self.avg_readout(one_graph, feat), self.max_readout(one_graph, feat)], dim=-1)
        feat = F.relu(self.lin1(feat))
        output = F.relu(self.lin3(feat))
        return output
    
class Word_Doc_Block_Simple(nn.Module):
    """
    Version: Gloabl Pool without classifier
    @refer: https://github.com/dmlc/dgl/tree/0.9.x/examples/pytorch/sagpool
    """
    def __init__(self, config):
        super(Word_Doc_Block_Simple, self).__init__()

        self.doc_linear_layer = nn.Linear(config.doc_in_dim, config.word_in_dim)
        self.dropout = config.feat_drops

        self.word_doc_gat_layer = nn.ModuleList()
        
        gat_in_dim = config.word_in_dim
        gat_out_dim = config.hid_dim
        for i in range(config.num_layers):
            self.word_doc_gat_layer.append(GATLayer(gat_in_dim, gat_out_dim,
                                                    config.num_heads, config.feat_drops,
                                                    config.attn_drops, config.alpha,
                                                    config.residuals, agg_modes=config.agg_modes))
            gat_in_dim = gat_out_dim * config.num_heads
        
        self.pool = OrientedPool(gat_in_dim, ratio=config.pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()
        
        self.lin1 = torch.nn.Linear(gat_in_dim * 2, config.hid_dim)
        self.lin2 = torch.nn.Linear(config.hid_dim,config.out_dim)

    def forward(self, word_doc_graph, doc_feat, word_feat, special_nodes_idx, device='cuda'):
        doc_feat = F.relu(self.doc_linear_layer(doc_feat))
        feat = torch.add(doc_feat, word_feat)

        one_graph = word_doc_graph.to(device)
        for gat_layer in self.word_doc_gat_layer:
            feat = gat_layer(one_graph, feat)
            feat = F.relu(feat) 
        pool_one_graph, pool_feat, _ = self.pool(one_graph, feat, need_guide = True, special_nodes_idx = special_nodes_idx)
        feat = torch.cat([self.max_readout(pool_one_graph, pool_feat), self.max_readout(one_graph, feat)], dim=-1)
        
        feat = F.relu(self.lin1(feat))
        output = F.relu(self.lin2(feat))
        return output
    

class SemanticAttention(nn.Module):
    """
    For Event Feature
    Attention Module
    @refer
    [1] Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction (WSDM2018)
    [2] Heterogeneous Graph Attention Network (WWW2019)
    """
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention,self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size,1, bias=False)
        )
    def forward(self, z, device):    
        w = self.project(z)            
        each_layer_weight = w.detach().cpu().numpy()

        beta = torch.softmax(w, dim=0)
        beta = beta.expand(z.shape[0], z.shape[1]) 
        return (beta * z).sum(0), each_layer_weight 

class CeleTrip_MultiLocation(nn.Module):
    def __init__(self, config):
        super(CeleTrip_MultiLocation, self).__init__()
        
        self.word_doc_block = Word_Doc_Block(config.subgraph_config)
        self.ea_layer = SemanticAttention(config.event_config.in_dim, config.event_config.hid_dim)
        self.ea_linear_layer= nn.Linear(config.event_config.in_dim, config.event_config.out_dim)

        self.en_layer = nn.Linear(config.entity_config.in_dim, config.entity_config.out_dim)
        self.en_dim = config.entity_config.out_dim
        
        self.mg_list = nn.ModuleList()
        mg_config = config.maingraph_config
        mg_in_dim, mg_out_dim = mg_config.in_dim, mg_config.out_dim
        for i in range(mg_config.num_layers):
            self.mg_list.append(GATLayer(mg_in_dim, mg_out_dim, mg_config.num_heads,
                                         mg_config.feat_drops, mg_config.attn_drops,
                                         mg_config.alpha, mg_config.residuals, mg_config.agg_modes))
            if mg_config.agg_modes == 'flatten':
                mg_in_dim = mg_out_dim * mg_config.num_heads
            else:
                mg_in_dim = mg_out_dim

        self.mg_classifier = nn.Linear(mg_out_dim*mg_config.num_heads, mg_config.outputs)
        self.crloss = nn.CrossEntropyLoss()
        
    def forward(self, mgraph, batched_subgraph, special_nodes_idx, 
                labels,
                event_feature, entity_features, location_latlon_list = None,
                device = 'cuda',
                loss_constrain = False):

        word_feat = batched_subgraph.ndata.pop("feat")
        word_feat = word_feat.to(device)
        subgraph_features = self.word_doc_block(batched_subgraph,word_feat, special_nodes_idx)

        event_result = []
        event_sentence_weight = []
        if len(event_feature)==0:
            eve_feat = []
        else:
            for eve in event_feature:
                
                eve = eve.to(device)
                eve = eve[:5]
                
                eve_feat,sentence_weight = self.ea_layer(eve.float(),device)
                eve_feat = eve_feat.unsqueeze(0)
                sentence_weight = sentence_weight.flatten()
                event_sentence_weight.append(sentence_weight)
                eve_feat = self.ea_linear_layer(eve_feat)
                eve_feat = F.relu(eve_feat)
                event_result.append(eve_feat)
            eve_feat = torch.cat(event_result)


        if len(entity_features)==0:
            en_feat = -1
        else:
            entity_feature = entity_features.to(device)
            en_feat = self.en_layer(entity_feature.float())
            en_feat = F.relu(en_feat)
        main_node_features = subgraph_features
        
        if len(eve_feat)!=0:
            main_node_features = torch.cat([main_node_features,eve_feat],dim=0)
        if type(en_feat)!=int:
            main_node_features = torch.cat([main_node_features,en_feat],dim=0)

        for mgal in self.mg_list:
            mgraph = mgraph.to(device)
            main_node_features = mgal(mgraph,main_node_features)
            main_node_features = F.relu(main_node_features)
        trip_tup = main_node_features[:batched_subgraph.batch_size] 
        
        outputs = self.mg_classifier(trip_tup)
        return outputs
    


from tqdm import tqdm
def train(model, device, train_loader, val_loader,test_loader, optimizer, criterion, scheduler, config):
    
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    
    best_acc = 0.0
    early_stop = 0
    
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        for batch, batch_dict in enumerate(train_loader):
            mgraph = batch_dict[0].to(device)
            batched_subgraph = dgl.batch(batch_dict[1]).to(device)
            special_nodes_idx = batch_dict[2].to(device)
            labels = batch_dict[3].to(device)
            event_features = batch_dict[4].to(device)
            entity_features = batch_dict[5].to(device)
            outputs = model(mgraph, batched_subgraph,
                            special_nodes_idx, labels,
                            event_features, entity_features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() / len(labels)
        
        scheduler.step()
        
        _, train_acc, _, _ = evaluate(train_loader, device, model, CRLoss)
        _, val_acc, _, _ = evaluate(val_loader, device, model, CRLoss)
        
        train_loss_list.append(train_loss / (batch + 1))
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        if epoch %10==0:
            _, _, test_labels, test_predict = evaluate(test_loader, device, model, CRLoss, show = True)
            rep_in_training = classification_report(test_labels,test_predict,digits=6,target_names=['non trip','trip'])
            logger.info('-'*20)
            logger.info('The repo in training : ')
            logger.info(rep_in_training)
            print('-'*20)
            print('The repo in training : ',epoch,rep_in_training)
        if val_acc >= best_acc:
            best_acc = val_acc
            print('Save the best model in one parameters!')
            torch.save(model.state_dict(), config.best_model_path)
            early_stop = 0
        else:
            early_stop += 1
        torch.cuda.empty_cache()
        if early_stop > config.patience:
            print('Stop epoch : ',epoch)
            break
    
    return train_loss_list, train_acc_list, val_acc_list


@torch.no_grad()
def evaluate(dataloader, device, model, criterion, show = False):
    model.eval()
    val_loss = 0
    total_predict = []
    total_labels = []
    for batch, batch_dict in enumerate(dataloader):

        
        mgraph = batch_dict[0].to(device)
        batched_subgraph = dgl.batch(batch_dict[1]).to(device)
        special_nodes_idx = batch_dict[2].to(device)
        labels = batch_dict[3].to(device)
        event_features = batch_dict[4].to(device)
        entity_features = batch_dict[5].to(device)
        
        

        outputs = model(mgraph, batched_subgraph,
                        special_nodes_idx, labels,
                        event_features, entity_features)

        loss = criterion(outputs, labels)
        # predict
        _, predicted = torch.max(outputs.data, 1)

        total_predict += list(predicted.cpu().numpy().reshape(-1))

        total_labels += list(labels.cpu().numpy().reshape(-1))

        val_loss += loss.item()
    score = accuracy_score(total_labels, total_predict)
    if show:
        rep = classification_report(total_labels, total_predict, digits=6,target_names=['non trip location','trip location'])
    return val_loss / (batch + 1), score, total_labels, total_predict

class WordDocConfig():
    """
    config for word-doc graph
    """
    def __init__(self):
        self.doc_in_dim = 100
        self.word_in_dim = 100
        self.hid_dim = 64
        self.hid_dim2 = 32

        self.outputs_dim = 2 
        self.out_dim = 64 


        self.num_layers = 3
        self.num_heads = 6
        self.feat_drops = 0.
        self.attn_drops = 0.
        self.residuals = True
        self.agg_modes = 'flatten'
        self.alpha = .2
        self.pool_ratio = 0.8
        
class EventConfig():
    """
    config for event feature
    """
    def __init__(self):
        self.in_dim = 100
        self.hid_dim = 64
        self.out_dim = 64
        
class EntityConfig():
    """
    config for entity feature
    """
    def __init__(self):
        self.in_dim = 50
        self.out_dim = 64
        
class MainGraphConfig():
    """
    config for main graph
    """
    def __init__(self):
        self.in_dim = 64
        self.out_dim = 32
        self.outputs = 2
        self.num_layers = 2
        self.num_heads = 4
        self.feat_drops = 0.
        self.attn_drops = 0.
        self.residuals = True
        self.agg_modes = 'flatten'
        self.alpha = .2    
        
class Config():
    def __init__(self):
        self.prefix_dir =  './graph_data/multi_location_5/'
        self.train_subgraph_feature_path= load_pkl('./graph_data/multi_location_5/train_rawdata')
        self.test_subgraph_feature_path  = load_pkl('./graph_data/multi_location_5/test_rawdata')
        self.train_maingraph_feature_path = load_pkl(self.prefix_dir + "/train_maingraph_feature.pkl")
        self.test_maingraph_feature_path = load_pkl(self.prefix_dir + "/test_maingraph_feature.pkl")
        self.maingraph_config = MainGraphConfig()
        self.subgraph_config = WordDocConfig()
        self.event_config = EventConfig()
        self.entity_config = EntityConfig()
        self.epoch =500
        self.device = 'cuda'
        self.step_size = 15
        self.lr_scheduler_gamma = 0.5
        self.patience = 15
        self.weight_decay = 5e-4
        self.learning_rate = 0.01
        self.best_model_path = './ModelResult/CeleTrip_Best_Model/CeleTrip_multi_loc_3.bin'
        self.save_result_path = './ModelResult/Best_Result/CeleTrip_multi_loc_3.csv' 
        self.save_prob_path = './ModelResult/CeleTrip_Pro/CeleTrip_multi_loc_3.pkl'
        self.final_best_model_path = './ModelResult/Final_Best_Model/CeleTrip_multi_loc_3.bin'
        self.log_file = './ModelResult/log/log_multi_loc_3.txt'
        self.whole_model = './ModelResult/Final_Best_Model/CeleTrip_multi_whole_3.pkl'
        
if __name__=='__main__':
       
    model_name = None
    running_mode = 'Whole'

    # init config
    config = Config()
    device = config.device

    # logger
    file_path = config.log_file
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    # create formatter
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # create file handler
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    # connect
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    train_dataset, test_dataset = None, None
    logger.info('==========Loading Data==========')

    train_dataset = MultiLocationDataset(config.train_maingraph_feature_path,
                                        config.train_subgraph_feature_path)

    test_dataset = MultiLocationDataset(config.test_maingraph_feature_path,
                                        config.test_subgraph_feature_path)

    
    logger.info('==========Starting Training==========')
    logger.info(len(train_dataset))
    logger.info(len(test_dataset))


    best_param_result = 0
    search_space = {
        'num_layers': [2,3],
        'out_dim': [16,64,32],
        'num_heads': [6,4,2],
        'dropout': [0],
        'attn_drop': [0],
        'alpha': [0.2],
        'residual': [True],
    }

    num_search_trials =10
    configure_generator = ParameterSampler(search_space, n_iter = num_search_trials, random_state=42)

    for i, configure in enumerate(configure_generator):

        logger.info(configure)
        train_idx, val_idx = train_test_split([i for i in range(len(train_dataset))], test_size = 0.2, random_state = 60)
        print('The length of train index : ',len(train_idx))
        print('The length of val index : ',len(val_idx))

        config.maingraph_config.num_layers = configure["num_layers"]
        config.maingraph_config.out_dim = configure["out_dim"]
        config.maingraph_config.num_heads = configure["num_heads"]
        config.maingraph_config.residuals = configure["residual"]
        config.maingraph_config.feat_drops = configure["dropout"]
        config.maingraph_config.attn_drops = configure["attn_drop"]
        config.maingraph_config.alpha = configure["alpha"]
        batch_size = None
        train_loader = DataLoader(
            train_dataset,
            sampler = train_idx,
            batch_size = batch_size, 
            drop_last = False)


        val_loader = DataLoader(
            train_dataset,
            sampler = val_idx,
            batch_size = batch_size,
            drop_last = False)

        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            drop_last = False)
        print('The length of dataloader : ',len(train_loader),len(val_loader),len(test_loader))
        model = CeleTrip_MultiLocation(config)
        if torch.cuda.is_available() == True:
            model = model.to(device)

        CRLoss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.lr_scheduler_gamma)

        train_loss_list, train_acc_list, val_acc_list = train(model, device, train_loader, val_loader,test_loader, optimizer, CRLoss, scheduler, config)

        model.load_state_dict(torch.load(config.best_model_path))

        test_loss, test_acc, test_labels, test_predict = evaluate(test_loader, device, model, CRLoss, show = True)
        one_rep = classification_report(test_labels, test_predict,digits=6, target_names=['non trip location','trip location'])
        logger.info('#'*20)
        logger.info('after epoch training : ')
        logger.info(one_rep)
        logger.info('#'*20)
        print('#'*20)
        print('after epoch training :',one_rep)
        print('#'*20)

        if test_acc > best_param_result:
            best_param_result = test_acc
            pred_result = pd.DataFrame(test_predict, columns=['pred'])
            pred_result['label'] = test_labels
            pred_result.to_csv(config.save_result_path, index = False, encoding='utf-8')
            best_cv_rep = classification_report(test_labels, test_predict, target_names=['non trip location','trip location'], output_dict=True)
            torch.save(model.state_dict(),config.final_best_model_path)
        torch.cuda.empty_cache() 
        print('Finishing :::::!!!!!')

