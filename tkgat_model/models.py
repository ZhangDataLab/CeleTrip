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


class SemanticAttention(nn.Module):
    def __init__(self,in_size,hidden_size=128):
        super(SemanticAttention,self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size,1,bias=False)
        )
    def forward(self,z,device):    
        # print('The size of z input : ',z.size())# 50 x 100
        w = self.project(z)            
        each_layer_weight = w.detach().cpu().numpy()
        # print('The weight of w : ',w[:5])

        beta = torch.softmax(w, dim=0)# 50 x 1
        # print('The size of beta : ',beta[:5])
        beta = beta.expand(z.shape[0],z.shape[1]) # 50x100
        # beta = torch.where(torch.isnan(beta),beta,(torch.tensor([0]).to(device)).float())
        # print('The size of beta weight : ',beta.size())# 50 x 100
        return (beta * z).sum(0),each_layer_weight# return 100

'''
outputs是输出的维度
mg_dim是主图的特征维度
mg_out是主图的隐藏层维度
'''
class HoTripGraph(nn.Module):
    def __init__(self,outputs,
                 mg_dim,mg_out,
                 sg_dim,sg_out,
                 na_dim,na_out,
                 ev_dim,ev_hid,ev_out,
                 en_dim,en_out,
                 num_layers,num_heads,feat_drops,attn_drops,alphas,residuals,agg_modes,activations=None):
        super(HoTripGraph,self).__init__()
        # 定义子图的图学习
        self.subg_list = nn.ModuleList()
        for i in range(num_layers):
            self.subg_list.append(GATLayer(sg_dim,sg_out,num_heads,feat_drops,attn_drops,alphas,residuals,agg_modes='flatten'))
            if agg_modes=='flatten':
                sg_dim = sg_out*num_heads
            else:
                sg_dim = sg_out
        self.sg_pooling = MaxPooling()
        self.sg_linear = nn.Linear(sg_out*num_heads,na_out)
        
        # 定义事件的attention
        self.ea_layer = SemanticAttention(ev_dim,ev_hid)
        self.ea_linear_layer= nn.Linear(ev_dim,ev_out)

        self.ev_freq_layer = nn.Linear(15,ev_out)
        self.ev_freq_con = nn.Linear(2*ev_out,ev_out)
        # 定义实体
        self.en_layer = nn.Linear(en_dim,en_out)
        self.en_compgcn = CompGCNCov(in_channels=50,out_channels=en_out)
        self.en_dim = en_out


        # 定义人名
        self.na_layer = nn.Linear(na_dim,na_out)
        
        # 定义主图层
        self.mg_list = nn.ModuleList()
        for i in range(num_layers):
            self.mg_list.append(GATLayer(mg_dim,mg_out,num_heads,feat_drops,attn_drops,alphas,residuals,agg_modes='flatten'))
            if agg_modes=='flatten':
                mg_dim = mg_out*num_heads
            else:
                mg_dim = mg_out
        # self.mg_classifier2 = nn.Linear(mg_out*num_heads*2,outputs)
        # 定义主图的分类器
        self.mg_classifier = nn.Linear(mg_out*num_heads,outputs)
        
    def forward(self,mgraph,subgraph_list,na_feature,event_feature,entity_features,event_freq_feature,entity_graph_list,entity_graph_features_list,device='cuda'):
        subgraph_features_list = []
        for subg in subgraph_list:
            one_subg = subg.to(device)
            feats = (subg.ndata['attr']).to(device)
            for i,sla in enumerate(self.subg_list):
                feats = F.relu(sla(one_subg,feats))
            feats = self.sg_pooling(one_subg,feats)
            feats = self.sg_linear(feats)
            subgraph_features_list.append(feats)
        subgraph_features = torch.stack(subgraph_features_list)

        #########################################################
        na_feat = F.relu(self.na_layer((na_feature.to(device)).float()))
        na_feat = na_feat.unsqueeze(0)
        #########################################################

        event_result = []
        event_sentence_weight = []
        if len(event_feature)==0:
            eve_feat = []
        else:
            ############################################################################
            for eve,eve_freq in zip(event_feature,event_freq_feature):
                # eve = eve.unsqueeze(0)
                eve = eve.to(device)
                eve = eve[:5]
                eve_feat,sentence_weight = self.ea_layer(eve.float(),device)
                eve_feat = eve_feat.unsqueeze(0)
                sentence_weight = sentence_weight.flatten()
                event_sentence_weight.append(sentence_weight)
                # print('eve_feat size ',eve_feat.size())
                eve_feat = self.ea_linear_layer(eve_feat)
                eve_feat = F.relu(eve_feat)

                #######################################################
                '''
                需要用到freq的话就加这段
                '''
                eve_freq = eve_freq.to(device)
                eve_freq = eve_freq.unsqueeze(0)
                eve_freq = self.ev_freq_layer(eve_freq.float())
                eve_feat = torch.cat([eve_feat,eve_freq],dim=1)
                eve_feat = F.relu(self.ev_freq_con(eve_feat))+eve_freq
                #######################################################

                event_result.append(eve_feat)
            # print('The length of event sentence : ',len(event_sentence_weight))
            eve_feat = torch.cat(event_result)
            ############################################################################

        ########################################################### 
        if len(entity_features)==0:
            en_feat = -1
        else:
            entity_feature = entity_features.to(device)
            en_feat = self.en_layer(entity_feature.float())
            en_feat = F.relu(en_feat)

            en_feat_list = []
            
            for ent,entf in zip(entity_graph_list,entity_graph_features_list):
                ent = ent.to(device)
                x_feat = torch.from_numpy(entf[0]).to(device)
                rel_feat = torch.from_numpy(entf[1]).to(device)
                x_edge_type = entf[2].to(device)
                x_edge_norm = entf[3].to(device)
                if len(x_edge_type)==0:
                    x_feat = torch.zeros((1,self.en_dim))
                    x_feat = x_feat.to(device)
                    en_feat_list.append(x_feat)
                    continue
                x_feat,rel_feat = self.en_compgcn(ent,x_feat,rel_feat,x_edge_type,x_edge_norm)
                one_ent_feat = F.relu(x_feat)
                en_feat_list.append(one_ent_feat[0].unsqueeze(0))

            # print('en_feat_list [0] ',en_feat_list[0].size())# 1 x 128
            en_feat = torch.cat(en_feat_list)
            
            # print('after cat : ',en_feat.size())# num_entity x 128
        ###########################################################

        '''
        合并子图和大图的特征
        '''
        main_node_features = torch.concat(subgraph_features_list)
        
        main_node_features = torch.cat([main_node_features,na_feat],dim=0)
        
        if len(eve_feat)!=0:
            main_node_features = torch.cat([main_node_features,eve_feat],dim=0)
        if type(en_feat)!=int:
            main_node_features = torch.cat([main_node_features,en_feat],dim=0)
        
        for mgal in self.mg_list:
#             print('1',main_node_features.size())
            mgraph = mgraph.to(device)
            main_node_features = mgal(mgraph,main_node_features)
#             print('1',main_node_features.size())
            main_node_features = F.relu(main_node_features)
        trip_tup = main_node_features[:len(subgraph_list)]

        #####################################################################
        '''
        需要合并人名和地点元组就用这段
        '''
        # name_feature_vector = main_node_features[len(subgraph_list)]
        # name_feature_vector = name_feature_vector.expand((trip_tup.size()))
        # trip_tup = torch.cat((trip_tup,name_feature_vector),dim=1)
        # trip_tup = self.mg_classifier2(trip_tup)
        #####################################################################
        
        trip_tup = self.mg_classifier(trip_tup)
        return trip_tup,event_sentence_weight




import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np

from torch.fft import irfft2
from torch.fft import rfft2
def rfft(x,d):
    t = rfft2(x,dim=(-d))
    return torch.stack((t.real,t.imag),-1)
def irfft(x,d,signal_sizes):
    return irfft2(torch.complex(x[:,:,0],x[:,:,1]),s=signal_sizes,dim=(-d))


class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr', num_base=-1,
                 num_rel=None):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.udf.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        # print('edge _type ',type(edge_type))
        # print('edge tpye ',edge_type)
        # print(self.rel[edge_type].size())
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)
        # 边只更新一次，集
        return self.act(x), torch.matmul(self.rel, self.w_rel)




###################################################################################
def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)# 10x100
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)# 1x10x100
        feature_list.append(feature)
    res = torch.cat(feature_list, 0).squeeze()
    return res

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

class EventSentence(nn.Module):
    def __init__(self,ev_in_size,ev_out_size,hidden_size=32):
        super(EventSentence,self).__init__()
        self.sent_weight = nn.Parameter(torch.Tensor(ev_in_size,hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1,hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(hidden_size,1))
        self.fc = nn.Linear(ev_in_size,ev_out_size)
        self._create_weights(mean=0.0,std=0.05)

    def _create_weights(self,mean=0.0,std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
    
    def forward(self,inputs):
        outputs = matrix_mul(inputs,self.sent_weight,self.sent_bias)
        res = matrix_mul(outputs,self.context_weight)
        outputs = res.permute(1,0)
        res = res.detach().cpu().numpy()
        outputs = F.softmax(outputs)
        outputs = element_wise_mul(inputs,outputs.permute(1,0)).squeeze(0)
        outputs = self.fc(outputs)
        return outputs,res
###################################################################################
       
