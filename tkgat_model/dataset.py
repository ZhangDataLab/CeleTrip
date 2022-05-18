# 构建相应的dataset
# 合并好这几个结果，然后丢进模型里边
import os
from random import shuffle
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
from torch.utils.data.sampler import SubsetRandomSampler

from gensim.models import Word2Vec, KeyedVectors
import gensim

import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


'''
main_graph_properties_path是主图的节点列表
main_date_graph_path 是主图的边路径

sub_graph_properties_path 是子图的节点列表
sub_graph_save_path 是子图的边路径

entity_save_path是实体的保存路径
event_save_path是事件的保存路径
'''
# main_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/properties/train_main_graph_properties.pkl'
# main_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_date_graph/'

# sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_train_properties/'
# sub_graph_save_path =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_train_graph/'

# entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_entity_features/'
# event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_event_features/'

# word_embeddings_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/wordvector/trip_corpus.model'

class SyntheticDataset(DGLDataset):
    def __init__(self,main_graph_properties_path,
                    main_date_graph_path,
                    sub_graph_properties_path,
                    sub_graph_save_path,
                    entity_save_path,
                    event_save_path,
                    word_embedding_path):
        self.main_graph_properties_path = main_graph_properties_path
        self.sub_graph_properties_path = sub_graph_properties_path
        self.sub_graph_save_path = sub_graph_save_path
        self.entity_save_path = entity_save_path
        self.event_save_path = event_save_path
        self.main_date_graph_path = main_date_graph_path
        self.word_embedding_path = word_embedding_path
        self.event_freq_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/properties/event_freq_features.pkl'
        self.ent_feature_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/ent_openke_feature.pkl'
        self.rel_feature_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/rel_openke_feature.pkl'
        self.ent_rel_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/ent_rel_dict.pkl'
        super().__init__(name='synthetic')
    
    def process(self):
        word_embeddings_dim =100
        self.dim_feats = word_embeddings_dim
        self.gclasses = 2
        
        ###################################
        '''
        读取训练好的Word2Vector
        '''
        model = gensim.models.Word2Vec.load('wordvector/trip_corpus.model')
        word_vocab = model.wv.vocab
        ###################################

        ###################################
        '''
        加载实体和关系
        '''
        ent_rel_dict=  load_pkl(self.ent_rel_path)
        ent_features= load_pkl(self.ent_feature_path)
        rel_features = load_pkl(self.rel_feature_path)

        ###################################



        '''
        先加载主图
        主图保存在
        '''
        self.graphs = []
        self.labels = []
        self.sub_graphs = []
        self.name_features = []
        self.entity_features = []
        self.event_features = []
        self.event_freq_features = []
        self.entity_graph_list =[]
        self.entity_graph_features = []
        
        main_graph_properties = load_pkl(self.main_graph_properties_path)
        main_cnt_list = main_graph_properties['graph_id'].tolist()
        main_graph_label_list = main_graph_properties['label'].tolist()
        main_graph_sub_graph_id= main_graph_properties['sub_graph_id'].tolist()
        main_graph_node_dict_list = main_graph_properties['node_dict'].tolist()
        main_graph_event_dict_list = main_graph_properties['event_dict'].tolist()
        main_graph_entity_dict_list = main_graph_properties['entity_dict'].tolist()
        test_cnt = 0
        for i,(mcnt,mlal,msgl,mndl,medl,mendl) in enumerate(zip(main_cnt_list, main_graph_label_list, main_graph_sub_graph_id, main_graph_node_dict_list, main_graph_event_dict_list, main_graph_entity_dict_list)):
            # 主图下所有行程三元组图的节点特征字典
            sub_graph_properties = load_pkl(self.sub_graph_properties_path+str(mcnt)+'.pkl')
            sub_graph_properties = sub_graph_properties.groupby(['graph_id'])
            sub_graph_properties = dict(list(sub_graph_properties))
            
            sub_graph_list = []# 保存子图的列表
            for j in msgl:# 读取日期图的小图，即行程子图
                sub_graph_edge = pd.read_csv(self.sub_graph_save_path+str(j)+'.csv')# 每个子图保存的路径，对应到生成时的cnt
                src = sub_graph_edge['src'].to_numpy()
                dst = sub_graph_edge['dst'].to_numpy()
                sub_num_nodes = len(sub_graph_edge['src'].unique())
                id_word_map = sub_graph_properties[j]['id_word'].tolist()[0]
                sub_graph_features = []
                for kid in id_word_map:# 保存行程三元组图的节点特征
                    sub_graph_features.append(model.wv[id_word_map[kid]])
                sub_graph_features = np.array(sub_graph_features)
#                 print('sub graph features : ',sub_graph_features.shape)
                try:
                    sub_graph = dgl.graph((src,dst),num_nodes=sub_num_nodes)
                except:
                    print(src)
                sub_graph = dgl.remove_self_loop(sub_graph)
                sub_graph = dgl.add_self_loop(sub_graph)
                sub_graph.ndata['attr'] = torch.Tensor(sub_graph_features)
                sub_graph_list.append(sub_graph)
                
                
            '''
            读取这一天下的实体和事件特征
            entity_features的维度是 entity_num x entity_dim
            event_features是dict，其中每个键是事件的index，值为 sen_num x sen_dim
            '''
            date_graph_entity_features = load_pkl(self.entity_save_path+str(mcnt)+'.pkl')
            sub_entity_graph_list = []
            sub_entity_feature_list = []
            for kv in mendl:
                rel_dict = ent_rel_dict[kv]
                src_list = rel_dict['src']
                dst_list = rel_dict['dst']
                rel_list = rel_dict['rel']
                rel_num = len(rel_list)
                rel_idx = 0
                node_idx = 1
                idx_dict= {}
                one_src_list = []
                one_dst_list = []
                rel_src_list = []
                rel_dst_list = []
                for i,(sidx,cidx) in enumerate(zip(src_list,dst_list)):
                    if cidx not in idx_dict:
                        idx_dict[cidx] = node_idx
                        node_idx+=1
                    one_src_list.append(0)
                    one_dst_list.append(idx_dict[cidx])
                    rel_src_list.append(rel_idx)
                    rel_dst_list.append(rel_idx+rel_num)
                    rel_idx+=1
#                 print('SRC List : ',one_src_list)
#                 print('DST List : ',one_dst_list)
#                 print('REL list : ',rel_src_list)
#                 print('REL list2 : ',rel_dst_list)
#                 print('The number of REL',rel_num)
                egraph = dgl.DGLGraph()
                egraph.add_nodes(node_idx)
                egraph.add_edges(one_src_list,one_dst_list)
                egraph.add_edges(one_src_list,one_dst_list)
                edge_type = torch.tensor(rel_src_list+rel_dst_list)
                in_deg = egraph.in_degrees(range(egraph.number_of_nodes())).float().numpy()
#                 print('In degree : ',in_deg)
                norm = in_deg**-0.5
                norm[np.isinf(norm)] = 0
                norm = torch.from_numpy(norm)
                egraph.ndata['xxx'] = norm
                egraph.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
                edge_norm = egraph.edata.pop('xxx').squeeze()
                x_feat = []
                x_feat.append(ent_features[sidx])
                for cidx in idx_dict:
                    x_feat.append(ent_features[cidx])
                x_feat = np.array(x_feat)
                one_rel_feat = []
                for ridx in rel_list:
                    one_rel_feat.append(rel_features[ridx])
                one_rel_feat+=one_rel_feat
                one_rel_feat  = np.array(one_rel_feat)
                
                sub_entity_graph_list.append(egraph)
                sub_entity_feature_list.append([x_feat,one_rel_feat,edge_type,edge_norm])
            self.entity_graph_list.append(sub_entity_graph_list)
            self.entity_graph_features.append(sub_entity_feature_list)
            
            date_graph_event_features = load_pkl(self.event_save_path+str(mcnt)+'.pkl')
            event_features = []
            for keve in date_graph_event_features:
                event_features.append(torch.from_numpy(date_graph_event_features[keve]))
            
            event_freq_features = []
            date_graph_event_freq_features = load_pkl(self.event_freq_path)
            for keve in date_graph_event_freq_features:
                event_freq_features.append(torch.from_numpy(date_graph_event_freq_features[keve]))


            person_name = mndl[len(msgl)]
            
            name_features = []
            person_name = person_name.split()
            for na in person_name:
                if na.lower() in word_vocab:
                    name_features.append(model.wv[na.lower()])
                else:
                    name_features.append(list(np.random.randn(100)))
            
            name_features = np.array(name_features).mean(axis=0)# 1x100
            
    
            #########################################################################################
            '''
            主图以同构图的方式连接
            msgl,mndl,medl,mendl  主图的子图列表，节点字典，事件节点，实体节点
            '''
            main_graph_edges = pd.read_csv(self.main_date_graph_path+str(mcnt)+'.csv')
            main_src = main_graph_edges['src'].to_numpy()
            main_dst = main_graph_edges['dst'].to_numpy()
            main_graph_node_number = len(main_graph_edges['src'].unique())
            
            # ------------------------------------------------------------------------------- #
            main_date_graph = dgl.graph((main_src,main_dst),num_nodes=main_graph_node_number)
            main_date_graph = dgl.remove_self_loop(main_date_graph)
            main_date_graph = dgl.add_self_loop(main_date_graph)
            # ------------------------------------------------------------------------------- #
            

            
            self.graphs.append(main_date_graph)
            self.labels.append(torch.LongTensor(mlal))
            self.sub_graphs.append(sub_graph_list)
            self.name_features.append(torch.from_numpy(name_features))
            self.entity_features.append(torch.from_numpy(date_graph_entity_features))
            self.event_features.append(event_features)
            self.event_freq_features.append(event_freq_features)
            test_cnt+=1
            # if test_cnt>10:
            #     break
#         return self.graphs,self.labels,self.event_features,self.entity_features,self.name_features
    
    def __getitem__(self,i):
        return self.graphs[i],self.labels[i],np.array(self.sub_graphs[i]),self.name_features[i],self.event_features[i],self.entity_features[i],self.event_freq_features[i],self.entity_graph_list[i],np.array(self.entity_graph_features[i])
    
    def __len__(self):
        return len(self.graphs)

if __name__ =='__main__':
    data = SyntheticDataset(main_graph_properties_path,main_date_graph_path,sub_graph_properties_path,sub_graph_save_path,entity_save_path,event_save_path,word_embeddings_path)