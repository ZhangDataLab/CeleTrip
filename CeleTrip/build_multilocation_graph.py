import os
from random import shuffle
import pandas as pd
import numpy as np
import math


import datetime
# 保存结果
import pickle

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
stopwords_list = list(set(stopwords.words('english')))
from nltk import sent_tokenize
stopwords_list.remove('in')
stopwords_list.remove('at')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.datasets import make_classification

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 记录一下这个图里边document的数量
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

import networkx as nx
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gensim.models import Word2Vec, KeyedVectors
import gensim
import pickle
word_embeddings_path = 'trip_w2v_model_new_lower.model'
# 句子的表示利用词向量的和
# 首先对所有的预料做训练
import re
from nltk.stem.snowball import SnowballStemmer
word_model = gensim.models.Word2Vec.load(word_embeddings_path)
word_vocab = word_model.wv.vocab

'''
生成每个文档对应的TFIDF向量
'''
def tfidf_learning(data,mf_num):
    if mf_num==-1:
        tfidf_vectorizer = TfidfVectorizer()
    else:
        tfidf_vectorizer = TfidfVectorizer(max_features = mf_num)
    data = tfidf_vectorizer.fit_transform(data)
    return data,tfidf_vectorizer


# def clean_sentence(review):
#     new_sen_list = []
#     for line in review:
#         words = line.lower().split()
# #         words = [w for w in words if not w in stopwords_list]

#         review_text = " ".join(words)

#         # Clean the text
#         review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
#         review_text = re.sub(r",", " ", review_text)
#         review_text = re.sub(r"\.", " ", review_text)
#         review_text = re.sub(r"!", " ", review_text)
#         review_text = re.sub(r"\?", " ", review_text)
#         review_text = re.sub(r"\s{2,}", " ", review_text)
#         new_sen_list.append(review_text)
#     return new_sen_list


def clean_sentence(sen_list):
    new_sen_list = []
    for sen in sen_list:
        if 'the hill 1625 k street nw suite 900 washington dc' in sen:
            continue
        words = sen.lower()
        words = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", words)
        words = re.sub(r",", " ", words)
        words = re.sub(r"\.", " ", words)
        words = re.sub(r"!", " ", words)
        words = re.sub(r"\?", " ", words)
        words = re.sub(r"\s{2,}", " ", words)
        words = words.split()
        words = [x for x in words if x not in stopwords_list]
        words = ' '.join(words)
        new_sen_list.append(words)
    return new_sen_list

def clean_article(article_list):
    new_article_list = []
    for article in article_list:
        one_article_list = []
        for sen in article:
            words = sen.lower()
            words = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", words)
            words = re.sub(r",", " ", words)
            words = re.sub(r"\.", " ", words)
            words = re.sub(r"!", " ", words)
            words = re.sub(r"\?", " ", words)
            words = re.sub(r"\s{2,}", " ", words)
            words = words.split()
            words = [x for x in words if x not in stopwords_list]
            words = ' '.join(words)
            one_article_list.append(words)
        new_article_list.append(one_article_list)
    return new_article_list



from collections import defaultdict
from math import log
'''
获得每个文档中的词，这些词要出现在word_vocab中
'''
def get_vocab(text_list):
    new_sen_list = []
    
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        words = [x.lower() for x in words]
        one_word_list = []
        for word in words:
            if word not in word_vocab:
                continue
            word_freq[word] += 1
            one_word_list.append(word)
        new_sen_list.append(' '.join(one_word_list))
    return word_freq,new_sen_list


'''
构建单词与文档的图，其中单词和文档以共现的方式构图，然后单词和单词以词共现的方式构图
sen_list是句子的列表
word_id_map是单词对应的index字典
id_word_map是index对应的单词字典
doc_id_map,id_doc_map文档对应的index
vocab是单词的词表
doc_list是文档的列表
window_size是词共现的窗口数量
'''
def build_edges(sen_list, word_id_map, id_word_map,doc_id_map,id_doc_map, vocab, doc_list,window_size=15):
    
    # ------------------------------------------------ # 
    '''
    计算sen_list中每个句子的单词词共现
    '''
    windows = []
    for doc_words in sen_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
            
    # 计算窗口中的单词
    word_pair_count = defaultdict(int)
    for window in windows:
        for i in range(1, len(window)):
            for j in range(i):# j表示i前面的单词
                word_i = window[i]
                word_j = window[j]# i前面的单词
                if word_j not in word_id_map or word_i not in word_id_map:
                    continue
                word_i_id = word_id_map[word_i]# 单词对应的index
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                # 保存对应的词对
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []
    num_window = len(windows)
    for word_id_pair, count in word_pair_count.items():
        i, j = word_id_pair[0], word_id_pair[1]# i表示前面的单词，j表示后面的单词
        row.append(i)# 前面的单词指向后面的单词，并且对应的index加上doc的数量
        col.append(j)# 前面的单词指向后面的单词，并且对应的index加上doc的数量
        weight.append(1)
        
    # 计算单词是否出现在该文档中
    doc_split_list = [x.split() for x in doc_list]
    for word in word_id_map:
        for doc,doc_word_split in zip(doc_list,doc_split_list):
            if word in doc_word_split:
                row.append(doc_id_map[doc])
                col.append(word_id_map[word])
                row.append(word_id_map[word])
                col.append(doc_id_map[doc])
                weight.append(1)
                weight.append(1)
    
    number_nodes = len(word_id_map)+ len(doc_list)
    for i in range(number_nodes):
        row.append(i)
        col.append(i)
        weight.append(1)
    return row,col,weight,number_nodes
def change_ent(train_df):
    ent_list = train_df['ent_list'].tolist()
    new_event_list = []
    new_ent_list = []
    for i,line in enumerate(ent_list):
        one_ent_list = []
        one_event_list = []
        for doc_ent in line:
            for ent in doc_ent:
                if ent[1]=='PERSON' or ent[1]=='ORG' or ent[1]=='FAC' or ent[1]=='NORP' or ent[1]=='PRODUCT':
                    one_ent_list.append(ent)
                elif ent[1]=='EVENT':
                    one_event_list.append(ent)
        new_ent_list.append(one_ent_list)
        new_event_list.append(one_event_list)
    return new_ent_list,new_event_list
from collections import defaultdict
from math import log
'''
构建文档和词的图，文档的id在词后边
'''
def build_text_graph(sen,doc, window_size,article_features_dict,word_model):
    word_freq,sen = get_vocab(sen)
    doc = list(set(doc))
    vocab = list(word_freq.keys())
    doc_features_list =[]
    doc_id_map = {doc:i+len(vocab) for i,doc in enumerate(doc)}
    id_doc_map = {i+len(vocab):doc for i,doc in enumerate(doc)}
    for one_doc in doc:
        doc_features_list.append(article_features_dict[one_doc])# 保存的是压缩矩阵的格式

    # 生成词图的特征
    word_id_map = {word:i for i,word in enumerate(vocab)}
    id_word_map = {i:word for i,word in enumerate(vocab)}

    word_features_list = []
    for word in word_id_map:
        word_features_list.append(word_model.wv[word])
    word_features_list = np.array(word_features_list)
    
    row,col,weight,number_nodes = build_edges(sen,word_id_map,id_word_map,doc_id_map,id_doc_map,vocab,doc,window_size)
    
    return row,col,weight,word_id_map,id_word_map,doc_id_map,id_doc_map,number_nodes,len(doc),word_features_list,doc_features_list

'''
构建图的框架
group表示传入的数据模式
related_event_sentence表示与事件相关的句子
config保存的路径的类
max_length表示事件的句子的数量
word_embedding_dim表示词嵌入的框架
'''
from tqdm import tqdm
def generate_graph(data,event_sentence_dict,config,article_features_dict,word_model,word_vocab,window_size = 15,max_length = 50,word_embedding_dim=100,is_Train=True):
    group = data.groupby(['name','date'])
    group = dict(list(group))
    cnt = 0
    main_cnt = 0
    properties_list = []
    sub_word_graph_dict  = {}
    sub_word_graph_property_dict ={}
    multi_location_graph_dict = {}
    for k in tqdm(group):
        date_is_ = k[1]
        row = []
        col = []
        main_weight = []
        one_df = group[k]
        label_list = one_df['label'].tolist()
        name_list = one_df['name'].tolist()
        date_list = one_df['date'].tolist()
        loc_list = one_df['location'].tolist()
        sen_list = one_df['clean_sen_list'].tolist()
        doc_list = one_df['article'].tolist()
        entity_list = one_df['entity'].tolist()
        event_list =one_df['event'].tolist()
        
        #--------------------- 地点的索引 ---------------------#
        '''
        range(start_loc,end_loc)是地点节点的范围
        '''
        node_dict = {}
        node_idx = 0
        start_loc = node_idx
        for i,loc in enumerate(loc_list):
            node_dict[node_idx] = loc # 索引——>地点
            node_idx+=1
        end_loc = node_idx
        #------------------------------------------------------#
    
        

        #-------------------- 事件index ---------------------#
        '''
        range(start_event,end_event)是事件的范围
        '''
        event_dict  ={}
        event_sen = {}
        start_event = node_idx
        for i,line in enumerate(event_list):
            if len(line)==0:
                continue
            for eve in line:# 每个eve都是[eve,'EVENT']的组合
                # 需要判断这个是否在event_article_dict中
                tup = (date_list[0],eve[0])
                eve = eve[0]
                if tup not in event_sentence_dict:
                    continue
                
                eve_sen = event_sentence_dict[tup]
 
                if tup not in event_dict:# 这个事件没有出现过
                    event_sen[tup] = eve_sen
                    event_dict[tup] = node_idx
                    node_dict[node_idx] = tup# 索引——>事件名
                    node_idx+=1
                    
                    row.append(i)
                    col.append(event_dict[tup])
                    row.append(event_dict[tup])
                    col.append(i)
                    
                    main_weight.append(1)
                    main_weight.append(1)
                    
                    # ------------------------- 事件出现在其他的文章句子 --------------------- #
 
                    for j,sen in enumerate(sen_list):
                        if i==j:
                            continue
                        this_sen_str = ' '.join(sen)
                        eve_split = eve.split()
                        eve_split = [x.lower() for x in eve_split]
                        eve_split  = ' '.join(eve_split)
                        if eve_split in this_sen_str:
#                             print('event in loc : ',eve_split,loc_list[j])
                            row.append(event_dict[tup])
                            col.append(j)
                            row.append(j)
                            col.append(event_dict[tup])
                            main_weight.append(1)
                            main_weight.append(1)
                    # ------------------------------------------------------------------------ #             
        end_event = node_idx
        #----------------------------------------------------------------------------------------------------------#
        
        #---------------------------------------------Entity List--------------------------------------------------#
        entity_dict = {}
        ent_list = []
        person_node_dict  ={}
        start_entity = node_idx
        for i,line in enumerate(entity_list):
            if len(line)==0:
                continue
            
            for ent in line:
                person_flag = 0
                if ent[1]=='PERSON':
                    person_flag=1
                
                ent = ent[0]
                if len(ent)<=1:
                    continue
                #-------------------- 判断这个实体是否在Wikidata中 -------------------- #
                if ent not in ent_rel_dict:
#                     print('not in wikidata : ',ent)
                    continue
                if ent in ent_rel_dict and len(ent_rel_dict[ent]['src'])==0:
                    continue
                #-------------------- 判断这个实体是否在Wikidata中 -------------------- #
                
                
                # ------------------- 判断这个实体是否指向已有的地名 ------------------ #
                exist_location_name = 0
                for loc in loc_list:
                    loc_split = loc.split()
                    loc_split = [x.lower().strip() for x in loc_split]
                    if ent.lower() in loc_split:
                        exist_location_name = 1
                        break
                if exist_location_name==1:
                    continue
                # ------------------- 判断这个实体是否指向已有的地名 ------------------ #
                
                
                # ------------------- 计算实体到事件和地点的连边 -------------------- #
                if ent not in entity_dict:
                    entity_dict[ent] = node_idx
                    ent_list.append(ent)
                    node_dict[node_idx] = ent # 索引 ----> 实体
                    if person_flag==1:
                        person_node_dict[node_idx] = ent
                    node_idx+=1

                    row.append(i)
                    col.append(entity_dict[ent])
                    row.append(entity_dict[ent])
                    col.append(i)

                    main_weight.append(1)
                    main_weight.append(1)         
        end_entity = node_idx
        #--------------------------------Entity List--------------------------------#

        
        # --------------------------------------------------------------------- #
        '''
        连接人物之间的边
        '''
        for i in person_node_dict:
            
            for j in person_node_dict:
                if i==j:
                    continue
                row.append(i)
                col.append(j)
        # --------------------------------------------------------------------- #
                
        
        # --------------------------------------------------------------------- #
        '''
        建立行程事件子图的词
        需要记录每一天的子图index
        date_graph_list和地点index是对应的
        '''
        date_graph_list = []
        sub_graph_properties = []
        for i,(sen,doc,loc) in enumerate(zip(sen_list,doc_list,loc_list)):# sen_list这里就有对应的人物和地点
            sub_row,sub_col,sub_weight,word_id_map,id_word_map,_,_,sub_number_nodes,doc_num,word_features,doc_features = build_text_graph(sen,doc,window_size,article_features_dict,word_model)
            sub_word_graph_dict[cnt] = {}
            sub_word_graph_dict[cnt]['row'] = sub_row
            sub_word_graph_dict[cnt]['col'] = sub_col
            sub_word_graph_dict[cnt]['weight']= sub_weight
            sub_word_graph_dict[cnt]['word_id_map'] = word_id_map
            sub_word_graph_dict[cnt]['id_word_map'] = id_word_map
            sub_word_graph_dict[cnt]['word_features'] = word_features
            sub_word_graph_dict[cnt]['doc_features'] = doc_features
            sub_word_graph_dict[cnt]['nodes_num'] = sub_number_nodes
            
            
            '''
            确定人物节点的下标和地点节点的下标
            '''
            person_name = name_list[0]
            person_name_split = person_name.split()
            person_name_split = [x.strip().lower() for x in person_name_split]
            person_word_id_map = {}
            for word in word_id_map:
                for pns in person_name_split:
                    if pns == word:
                        person_word_id_map[word_id_map[word]] = word
            
            loc_name_split = loc.split()
            loc_name_split = [x.lower() for x in loc_name_split]
            location_word_id_map = {}
            for word in word_id_map:
                for lns in loc_name_split:
                    if lns==word:
                        location_word_id_map[word_id_map[word]]=word
            sub_word_graph_dict[cnt]['person_id_word_map'] = person_word_id_map
            sub_word_graph_dict[cnt]['location_id_word_map'] = location_word_id_map
            
            date_graph_list.append(cnt)
            cnt+=1
        # --------------------------------------------------------------------- #
        
        
        
        # --------------------------------------------------------------------- #
        '''
        得到事件相关的句子
        eve_sen_dict是事件的句子特征矩阵
        '''
        eve_sen_dict = {}
        for i in range(start_event,end_event):
            tup = node_dict[i]
            eve_name= tup[1]
            eve_sen = event_sentence_dict[tup]
            eve_sen = eve_sen[:max_length]
            eve_sen_matrix = np.zeros((max_length,word_embedding_dim))

            for j,one_sen in enumerate(eve_sen):
                one_sen = [word_model.wv[x] for x in one_sen if x.lower() in word_vocab]
                one_sen = np.array(one_sen)
                one_sen_mean = one_sen.mean(axis=0)# 对所有词做加全求平均
                eve_sen_matrix[j,:] = one_sen_mean
            eve_sen_dict[i] = eve_sen_matrix
        # --------------------------------------------------------------------- #
        

        
        # --------------------------------------------------------------------- #
        '''
        获得实体的特征嵌入
        entity_features对应的就是实体节点的嵌入
        '''
        entity_features = np.zeros((len(entity_dict),50))
        for i,kv in enumerate(ent_list):
            
            one_entity_result = ent_rel_dict[kv]['src'][0]
            one_entity_result = ent_features[one_entity_result]
            entity_features[i,:] = one_entity_result
        # --------------------------------------------------------------------- #

        
        # --------------------------------------------------------------------- #
        '''
        保存到相应的文件夹下，先保存主图
        包括主图的实体特征，事件特征，图的边和节点特征
        main_cnt是日期图的编号
        label_list是标签的列表
        date_graph_list是人物行程元组的，用来从路径提取这个日期下的行程图，存的是cnt
        node_dict是人物三元组、人名、事件和实体的节点字典
        event_dict是事件的字典，对应的特征矩阵
        entity_dict是对应的实体矩阵，对应字典的实体
        '''
        if is_Train==True:
            save_pkl(config.train_entity_save_path+str(main_cnt)+'.pkl',entity_features)
            save_pkl(config.train_event_save_path+str(main_cnt)+'.pkl',eve_sen_dict)
            save_pkl(config.train_event_sent_save_path+str(main_cnt)+'.pkl',event_sen)
            multi_location_graph_dict[main_cnt] = {}
            multi_location_graph_dict[main_cnt]['row'] = row
            multi_location_graph_dict[main_cnt]['col'] = col
            multi_location_graph_dict[main_cnt]['weight'] = main_weight
            multi_location_graph_dict[main_cnt]['nodes_num'] = len(node_dict)

        else:
            save_pkl(config.test_entity_save_path+str(main_cnt)+'.pkl',entity_features)
            save_pkl(config.test_event_save_path+str(main_cnt)+'.pkl',eve_sen_dict)
            save_pkl(config.test_event_sent_save_path+str(main_cnt)+'.pkl',event_sen)        
            multi_location_graph_dict[main_cnt] = {}
            multi_location_graph_dict[main_cnt]['row'] = row
            multi_location_graph_dict[main_cnt]['col'] = col
            multi_location_graph_dict[main_cnt]['weight'] = main_weight
            multi_location_graph_dict[main_cnt]['nodes_num'] = len(node_dict)
        properties_list.append([main_cnt,np.array(label_list),date_graph_list,node_dict,event_dict,entity_dict,date_is_,person_node_dict,name_list[0]])
        # --------------------------------------------------------------------- #
        
        main_cnt+=1
#         if main_cnt>=50:
#             break
    if is_Train==True:
        # 保存semantic graph的节点特征
        properties_list = pd.DataFrame(properties_list,columns=['graph_id','label','sub_graph_id','node_dict','event_dict','entity_dict','date','person_dict','person'])
        save_pkl(config.train_multi_location_graph_properties,properties_list)
        
        save_pkl(config.train_word_graph,sub_word_graph_dict)
        save_pkl(config.train_multi_location_graph,multi_location_graph_dict)
    else:
        properties_list = pd.DataFrame(properties_list,columns=['graph_id','label','sub_graph_id','node_dict','event_dict','entity_dict','date','person_dict','person'])
        save_pkl(config.test_multi_location_graph_properties,properties_list)
        
        save_pkl(config.test_word_graph,sub_word_graph_dict)
        save_pkl(config.test_multi_location_graph,multi_location_graph_dict)


train_df = load_pkl('./data/train_data.pkl')
test_df = load_pkl('./data/test_data.pkl')

train_df['article'] = train_df['article'].map(clean_article)
test_df['article'] = test_df['article'].map(clean_article)
train_df['clean_sen_list'] = train_df['clean_sen_list'].map(clean_sentence)
test_df['clean_sen_list'] = test_df['clean_sen_list'].map(clean_sentence)
train_df['label'] = train_df['label'].map(int)
test_df['label'] = test_df['label'].map(int)

def merge_article(x):
    new_line = [' '.join(xx) for xx in x]
    return new_line
train_df['article'] = train_df['article'].map(merge_article)
test_df['article'] = test_df['article'].map(merge_article)



# ------------------------------------------------- #
'''
生成article对应的TFIDF向量
通过article_features_dict保存每个文档对应的TFIDF向量
'''
train_article_list = train_df['article'].tolist()
test_article_list = test_df['article'].tolist()
article_list = []
for line in train_article_list:
    article_list+=line
for line in test_article_list:
    article_list+=line
article_list = list(set(article_list))
len(article_list)

tfidf_data,_ = tfidf_learning(article_list,100)


article_features_dict = {}
for i,art in enumerate(article_list):
#     article_features_dict[art] = tfidf_data[i].toarray()# 1 x 10000
    article_features_dict[art] = tfidf_data[i]
del tfidf_data,article_list
# ------------------------------------------------- #


# ------------------------------------------------- #
'''
加载事件句子，实体关系的数据
将train和test的实体进行拆分，变成实体和事件
保存的格式是 [实体，实体类型]
'''
event_sentence_dict = load_pkl('./Clean_data/event_sentence_dict.pkl')
ent_rel_dict = load_pkl('./data/ent_rel_dict.pkl')
ent_features = load_pkl('./data/ent_openke_feature.pkl')
rel_features = load_pkl('./data/rel_openke_feature.pkl')
train_ent_list,train_event_list = change_ent(train_df)
test_ent_list,test_event_list = change_ent(test_df)

train_df['entity'] = train_ent_list
train_df['event'] = train_event_list
test_df['entity'] = test_ent_list
test_df['event'] = test_event_list
# ------------------------------------------------- #



#####################################################################################
class Config:
    window_size = 15
    save_path = './graph_data/multi_location_graph_'+str(window_size)+'/'
    
    train_multi_location_graph  = save_path + 'train_multi_location_graph.pkl'
    test_multi_location_graph = save_path + 'test_multi_location_graph.pkl'
    train_multi_location_graph_properties = save_path + 'train_multi_location_graph_properties.pkl'
    test_multi_location_graph_properties = save_path + 'test_multi_location_graph_properties.pkl'
    
    train_word_graph =save_path + 'train_word_graph.pkl'
    test_word_graph = save_path +'test_word_graph.pkl'

    train_entity_save_path = save_path + 'train_entity_path/'
    train_event_save_path = save_path + 'train_event_path/'
    train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    test_entity_save_path = save_path + 'test_entity_path/'
    test_event_save_path = save_path + 'test_event_path/'
    test_event_sent_save_path = save_path + 'test_event_sen_path/'
#####################################################################################

config = Config()
# 保存的主路径
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
    
    
# 实体子图的保存路径保存
if not os.path.exists(config.train_entity_save_path):
    os.makedirs(config.train_entity_save_path)
    
if not os.path.exists(config.test_entity_save_path):
    os.makedirs(config.test_entity_save_path)

# 事件句子保存路径保存
if not os.path.exists(config.train_event_save_path):
    os.makedirs(config.train_event_save_path)
    
if not os.path.exists(config.test_event_save_path):
    os.makedirs(config.test_event_save_path)

# 事件句子的权重保存
if not os.path.exists(config.train_event_sent_save_path):
    os.makedirs(config.train_event_sent_save_path)
    
if not os.path.exists(config.test_event_sent_save_path):
    os.makedirs(config.test_event_sent_save_path)
    
print('###### Start Train ######')
train_df.head()
generate_graph(train_df,event_sentence_dict,config,article_features_dict,word_model,word_vocab,window_size =config.window_size,max_length = 50,word_embedding_dim=100,is_Train=True)
print('###### Finish Train ######')

print('###### Start Train ######')
test_df.head()
generate_graph(test_df,event_sentence_dict,config,article_features_dict,word_model,word_vocab,window_size = config.window_size,max_length = 50,word_embedding_dim=100,is_Train=False)
print('###### Finish Train ######')

# ------------------------------------------------------------------------------------------------------------ #

def generate_maingraph_info(main_cnt_to_name_date, multi_location_graph, multi_location_graph_properties):
    """
    将原始的三个文件转为一个 xxx_maingraph_info.pkl
    xxx_maingraph_info = {
      "meta_data": {
        "maingraph_id2key_map": maingraph_id2key_map,
        "maingraph_properties": maingraph_properties [DataFrame]
      },
      "maingraph_info": maingraph_info
    }
    
    maingraph_info = {
      main_id : {
        'row': [0, 1, 0, 2],
        'col': [1, 0, 2, 0],
        'weight': [1, 1, 1, 1],
        'nodes_num': 3,
        'entity_features': tensor(),
        'event_features': dict(), // 50*100, key为事件节点的id
        'event_sentences': dict()
      }
    }
    
    """
    maingraph_info = {
        "meta_data": {
            "maingraph_id2key_map": main_cnt_to_name_date,
            "maingraph_properties": multi_location_graph_properties
        }, 
        "maingraph_info": multi_location_graph
    }
    return maingraph_info

prefix_dir = config.save_path
train_main_cnt_to_name_date = load_pkl(config.train_main_cnt_to_name_date_path)
train_multi_location_graph = load_pkl(config.train_multi_location_graph)
train_multi_location_graph_properties = load_pkl(prefix_dir + "train_multi_location_graph_properties.pkl")
test_main_cnt_to_name_date = load_pkl(config.test_main_cnt_to_name_date_path)
test_multi_location_graph = load_pkl(config.test_multi_location_graph)
test_multi_location_graph_properties = load_pkl(prefix_dir + "test_multi_location_graph_properties.pkl")

train_maingraph_info = generate_maingraph_info(train_main_cnt_to_name_date, train_multi_location_graph, train_multi_location_graph_properties)
# save it
save_pkl(prefix_dir + "train_maingraph_info.pkl", train_maingraph_info)
# three files to maingraph_info.pkl
test_maingraph_info = generate_maingraph_info(test_main_cnt_to_name_date, test_multi_location_graph, test_multi_location_graph_properties)
# save it
save_pkl(prefix_dir + "test_maingraph_info.pkl", test_maingraph_info)


# change train_subgraph_info.pkl to dgl graph test_subgraph_feature.pkl
def generate_dgl_maingraph(raw_dataset):
    """
    从 dict 类型的数据生成主图的dgl同构图数据
    xxx_maingraph_info = {
      "meta_data": {
        "maingraph_id2key_map": maingraph_id2key_map,
        "maingraph_properties": maingraph_properties [DataFrame]
      },
      "maingraph_info": maingraph_info
    }
    对 subgraphs_info 进行处理，将原 graph info list 处理为
    [batched_graph, special_nodes_list(tensor), label_list(tensor)]
    """
    maingraph_info = raw_dataset['maingraph_info']
    
    for main_id in maingraph_info:

        maingraph = maingraph_info[main_id]
        try:
            # construct graph
            g = dgl.graph((torch.tensor(maingraph["row"]), torch.tensor(maingraph["col"])), num_nodes = maingraph['nodes_num'])
            g = dgl.to_bidirected(g)# 双向边
            g = dgl.remove_self_loop(g) # remove original self-loop, avoiding add twice
            g = dgl.add_self_loop(g) # self-loop for GAT
            # entity feature: numpy->tensor (N,50)
            entity_features = torch.from_numpy(maingraph["entity_features"].astype('float32'))
            # event feature: dict->tensor (N,50,100)
            event_features = numpy.array([maingraph["event_features"][key] for key in maingraph["event_features"]])
            event_features = torch.from_numpy(event_features.astype('float32'))

            maingraph_info[main_id] = [g, entity_features, event_features]
        except:
            print(main_id)

        # 原地修改
    raw_dataset["maingraph_info"] = maingraph_info

# train
_ = generate_dgl_maingraph(train_maingraph_info)
# test
_ = generate_dgl_maingraph(test_maingraph_info)

# 从metapath中去掉主图的dataframe
train_maingraph_info["meta_data"].pop('maingraph_properties')
test_maingraph_info["meta_data"].pop('maingraph_properties')

save_pkl(prefix_dir + "train_maingraph_feature.pkl", train_maingraph_info)
save_pkl(prefix_dir + "test_maingraph_feature.pkl", test_maingraph_info)
# ------------------------------------------------------------------------------------------------------------ #




