import os
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling, SumPooling
import pandas as pd
import numpy as np
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling, SumPooling
import networkx as nx
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.nn.pytorch import GATConv
'''
传入主图mg_dim
子图列表sg_dim
人名特征na_dim
事件特征ev_dim
实体特征en_dim
'''
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_modes='flatten'):
        super(GATLayer, self).__init__()
        # 
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

class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(GCNLayer,self).__init__()
        self.gcn_conv = GraphConv(in_feats,out_feats)

    def forward(self,bg,feats):
        feats = self.gcn_conv(bg,feats)
        return feats

class CeleTrip_Word_Doc(nn.Module):
    def __init__(self,config):
        super(CeleTrip_Word_Doc,self).__init__()
        
        self.doc_linear_layer = nn.Linear(config.doc_in_dim,config.hid_dim)
        self.word_linear_layer = nn.Linear(config.word_in_dim,config.hid_dim)

        self.word_doc_gat_layer = nn.ModuleList()
        gat_in_dim = config.hid_dim
        gat_out_dim = config.hid_dim
        
        for i in range(config.num_layers):
            self.word_doc_gat_layer.append(GATLayer(gat_in_dim,gat_out_dim,config.num_heads,config.feat_drops,config.attn_drops,config.alpha,config.residuals,agg_modes=config.agg_modes))
            gat_in_dim = gat_out_dim * config.num_heads
        
        self.second_linear = nn.Linear(gat_in_dim,config.hid_dim2)
        self.classifier = nn.Linear(config.hid_dim2,config.outputs_dim)
        self.graph_pooling = MaxPooling()

    def forward(self,word_doc_graph,doc_feat,word_feat,device='cuda'):
        doc_feat = F.relu(self.doc_linear_layer(doc_feat))
        word_feat = F.relu(self.word_linear_layer(word_feat))
        feat = torch.concat([doc_feat,word_feat],dim=0)
        
        one_graph =dgl.to_bidirected(word_doc_graph)
        one_graph = dgl.remove_self_loop(one_graph)
        one_graph = dgl.add_self_loop(one_graph)
        one_graph = one_graph.to(device)
        for i,gat_layer in enumerate(self.word_doc_gat_layer):
            feat = gat_layer(one_graph,feat)
            feat = F.relu(feat)
        feat = self.graph_pooling(one_graph,feat)
        feat = F.relu(self.second_linear(feat))
        output = self.classifier(feat)
        return output
