import pandas as pd
import os
import numpy as np
import re
import time
import datetime
from torch.utils import data
import tqdm
import random
import dgl
import dgl.data
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import math
from dgl.nn import GraphConv
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.model_selection import ParameterSampler

from models.models import *
from dataset.dataset import *

from matplotlib import pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING']='0'

import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Config:
    device = 'cuda:0'
    # ######################################################################################
    # train_main_properties =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/properties/train_main_graph_properties.pkl'
    # test_main_properties = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/properties/test_main_graph_properties.pkl'
    
    # train_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_date_graph/'
    # test_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/test_date_graph/'

    # train_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_train_graph/'
    # test_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_test_graph/'

    # train_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_train_properties/'
    # test_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/sub_test_properties/'

    # train_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_entity_features/'
    # train_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/train_event_features/'

    # test_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/test_entity_features/'
    # test_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/test_event_features/'
    
    # best_model_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/best_models/best_model_1.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/pic/train_loss_fig_1.png'
    # log_file = './log/log_test_1.txt'
    # event_file = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/event_sentence_weight/event_sentence_weight_1.pkl'
    # ########################################################################################

    # ######################################################################################
    # train_main_properties =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/properties/train_main_graph_properties.pkl'
    # test_main_properties = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/properties/test_main_graph_properties.pkl'
    
    # train_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/train_date_graph/'
    # test_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/test_date_graph/'

    # train_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/sub_train_graph/'
    # test_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/sub_test_graph/'

    # train_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/sub_train_properties/'
    # test_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/sub_test_properties/'

    # train_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/train_entity_features/'
    # train_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/train_event_features/'

    # test_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/test_entity_features/'
    # test_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data2/test_event_features/'
    # best_model_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/best_models/best_model_subg.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/pic/train_loss_subg_fig.png'
    # log_file = './log/log_test_2.txt'
    # event_file = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/event_sentence_weight/event_sentence_weight_subg.pkl'
    # ########################################################################################

    ######################################################################################
    # train_main_properties =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/properties/train_main_graph_properties.pkl'
    # test_main_properties = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/properties/test_main_graph_properties.pkl'
    
    # train_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/train_date_graph/'
    # test_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/test_date_graph/'

    # train_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/sub_train_graph/'
    # test_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/sub_test_graph/'

    # train_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/sub_train_properties/'
    # test_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/sub_test_properties/'

    # train_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/train_entity_features/'
    # train_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/train_event_features/'

    # test_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/test_entity_features/'
    # test_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data3/test_event_features/'
    # best_model_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/best_models/best_model_3.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/pic/train_loss_subg_fig_3.png'
    # log_file = './log/log_test_3.txt'
    # event_file = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/event_sentence_weight/event_sentence_weight_subg_3.pkl'
    ########################################################################################

    ######################################################################################
    # train_main_properties =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/properties/train_main_graph_properties.pkl'
    # test_main_properties = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/properties/test_main_graph_properties.pkl'
    
    # train_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/train_date_graph/'
    # test_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/test_date_graph/'

    # train_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/sub_train_graph/'
    # test_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/sub_test_graph/'

    # train_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/sub_train_properties/'
    # test_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/sub_test_properties/'

    # train_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/train_entity_features/'
    # train_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/train_event_features/'

    # test_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/test_entity_features/'
    # test_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data4/test_event_features/'
    # best_model_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/best_models/best_model_4.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/pic/train_loss_subg_fig_4.png'
    # log_file = './log/log_test_4.txt'
    # event_file = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/event_sentence_weight/event_sentence_weight_subg_4.pkl'
    ########################################################################################

    ########################################################################################
    train_main_properties =  '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/properties/train_main_graph_properties.pkl'
    test_main_properties = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/properties/test_main_graph_properties.pkl'
    
    train_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/train_date_graph/'
    test_date_graph_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/test_date_graph/'

    train_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/sub_train_graph/'
    test_sub_graph_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/sub_test_graph/'

    train_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/sub_train_properties/'
    test_sub_graph_properties_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/sub_test_properties/'

    train_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/train_entity_features/'
    train_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/train_event_features/'

    test_entity_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/test_entity_features/'
    test_event_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data5/test_event_features/'
    
    best_model_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/best_models/best_model_5_5_5_5.bin'
    pic_save_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/pic/train_loss_fig_5_5_5_5.png'
    log_file = './log/log_test_5_5_5.txt'
    event_file = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/event_sentence_weight/event_sentence_weight_5_5_5_5.pkl'
    ########################################################################################

    word_embeddings_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/wordvector/trip_corpus.model'
    # graph 3 2 256 64 100 256 100 256 100 256 256 100 256 2 8 0 0 0.2
    output_dim = 2
    mg_in_dim = 128
    mg_out_dim=128
    sg_in_dim = 100
    sg_out_dim = 128
    na_in_dim=100
    na_out_dim = 128
    ev_in_dim = 100
    ev_hid_dim = 128
    ev_out_dim = 128
    en_in_dim = 100
    en_out_dim = 128
    num_layers = 2
    num_heads = 8
    feat_drops = 0.
    attn_drops = 0.
    alphas = 0.2
    residuals = False
    agg_modes = 'flatten'
    epoch=1000
    learning_rate = 0.0005
    step_size = 50
    lr_scheduler_gamma=0.5
    patience = 10
    weight_decay = 5e-4

config = Config()
device = config.device

# if torch.cuda.is_available():
#     device = config.device
# else:
#     device = 'cpu'

import logging
file_path = config.log_file
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())# 打印时间
fh = logging.FileHandler(file_path)# 创建日志文件
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

random_seed = 42
def seed_everything(seed=random_seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(random_seed)

def train(model,data,train_sampler,optimizer,criterion,epoch):
    model.train()
    running_loss = 0
    total_iters = len(train_dataloader)
    for i,idx in enumerate(train_sampler):
        batch = data[idx]
        labels = batch[1].to(device)
        main_graph = batch[0].to(device)
        sub_graph_list = batch[2]
        name_feature = batch[3].to(device)
        event_feature=  batch[4]
        entity_feature = batch[5].to(device)
        event_freq_feature = batch[6]# 记录事件的前后频率
        entity_graph_list = batch[7]
        entity_graph_features_list = batch[8]
        outputs,_ = model(main_graph,sub_graph_list,name_feature,event_feature,entity_feature,event_freq_feature,entity_graph_list,entity_graph_features_list,device)

        loss = criterion(outputs,labels)
        running_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),norm_type=2,max_norm=20)
        optimizer.step()
    running_loss = running_loss / len(train_sampler)
    return running_loss


def eval_net(model,data,index_sampler,criterion,show=False):    
    model.eval()

    total = len(index_sampler)
    total_loss  = 0
    total_correct = 0
    total_predict =[]
    total_labels = []
    event_sentences_weight = []

    for i,idx in enumerate(index_sampler):

        batch = data[idx]
        labels = batch[1].to(device)
        main_graph = batch[0].to(device)
        sub_graph_list = batch[2]
        name_feature = batch[3].to(device)
        event_feature=  batch[4]
        entity_feature = batch[5].to(device)
        event_freq_feature = batch[6]
        entity_graph_list = batch[7]
        entity_graph_features_list= batch[8]
        outputs,eve_sen_weight = model(main_graph,sub_graph_list,name_feature,event_feature,entity_feature,event_freq_feature,entity_graph_list,entity_graph_features_list,device)
        loss = criterion(outputs,labels)

        event_sentences_weight.append(eve_sen_weight)
        _,predicted = torch.max(outputs.data,1)
        predicted = list(predicted.cpu().numpy())
        total_predict+=predicted
        label_list = list(labels.cpu().numpy())
        total_labels+=label_list
        total_loss+=loss.item()
    model.train()
    avg_loss = total_loss / len(index_sampler)
    score = accuracy_score(total_labels, total_predict)# 计算验证集的准确率
    precision = precision_score(total_labels, total_predict,average='macro')
    recall = recall_score(total_labels, total_predict,average='macro')
    f1 = f1_score(total_labels, total_predict,average='macro')
    if show:
        print('The length of labels : ',len(total_labels))
        print('The length of prediction labels : ',len(total_predict))
        score = accuracy_score(total_labels, total_predict)# 计算验证集的准确率
        precision = precision_score(total_labels, total_predict,average='macro')
        recall = recall_score(total_labels, total_predict,average='macro')
        f1 = f1_score(total_labels, total_predict,average='macro')
        print('Final Accuracy : ',score)
        print('Final Precision : ',precision)
        print('Final Recall : ',recall)
        print('Final f1_score : ',f1)
        rep = classification_report(total_labels, total_predict,digits=6,target_names=['non trip location','trip location'])
        print('Report :\n',rep)
        logger.info('Classification Report : ')
        logger.info(rep)
    return avg_loss,total_predict,total_labels,event_sentences_weight,score,precision,recall,f1

if __name__=='__main__':
    print('Start Train')
    train_dataloader = SyntheticDataset(config.train_main_properties,
                                        config.train_date_graph_path,
                                        config.train_sub_graph_properties_path,
                                        config.train_sub_graph_save_path,
                                        config.train_entity_save_path,
                                        config.train_event_save_path,
                                        config.word_embeddings_path
                                        )
    print('Start Test')
    test_dataloader = SyntheticDataset(config.test_main_properties,
                                        config.test_date_graph_path,
                                        config.test_sub_graph_properties_path,
                                        config.test_sub_graph_save_path,
                                        config.test_entity_save_path,
                                        config.test_event_save_path,
                                        config.word_embeddings_path
                                        )
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    best_predict = 0
    max_epoch = config.epoch
    cv_num = 20

    for cv in range(cv_num):
        CRLoss = nn.CrossEntropyLoss()
        model = HoTripGraph(config.output_dim,
                            config.mg_in_dim,
                            config.mg_out_dim,
                            config.sg_in_dim,
                            config.sg_out_dim,
                            config.na_in_dim,
                            config.na_out_dim,
                            config.ev_in_dim,
                            config.ev_hid_dim,
                            config.ev_out_dim,
                            config.en_in_dim,
                            config.en_out_dim,
                            config.num_layers,
                            config.num_heads,
                            config.feat_drops,
                            config.attn_drops,
                            config.alphas,
                            config.residuals,
                            agg_modes=config.agg_modes)
        if torch.cuda.is_available()==True:
            model = model.to(config.device)
        print(torch.cuda.is_available())
        optimizer = optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=config.step_size,gamma=config.lr_scheduler_gamma)
        
    
        num_samples = len(train_dataloader)
        num_train = int(num_samples*0.8)
        # 固定train, val设置随机的index
        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        val_sampler = SubsetRandomSampler(torch.arange(num_train,num_samples))

        best_acc = 0.0
        early_stop = 0
        train_loss_list = []
        val_loss_list = []
        for ep in range(max_epoch):
            train_loss = train(model,train_dataloader,train_sampler,optimizer,CRLoss,ep)
        #     break
        # break
            scheduler.step()
            train_loss,train_predict,train_labels,train_event_sentences_weight,train_score,train_precision,train_recall,train_f1 = eval_net(model,train_dataloader,train_sampler,CRLoss)
            val_loss,val_predict,val_labels,val_event_sentences_weight,val_score,val_precision,val_recall,val_f1 = eval_net(model,train_dataloader,val_sampler,CRLoss)
            train_loss_list.append(train_score)
            val_loss_list.append(val_score)
            print('Train Loss : ',train_loss)
            print('Val Accuracy : ',val_score)
            if val_score>=best_acc:
                best_acc = val_score
                torch.save(model.state_dict(),config.best_model_path)
                print('###### Save the best model #######')
                early_stop = 0
            else:
                early_stop+=1
            torch.cuda.empty_cache()
            if early_stop>config.patience:
                break

        plt.figure(figsize=(10,6),dpi=300)
        plt.plot(train_loss_list,label='train')
        plt.plot(val_loss_list,label='val')
        plt.legend()
        plt.savefig(config.pic_save_path,bbox_inches='tight')

        model.load_state_dict(torch.load(config.best_model_path))
        test_sampler = list(range(len(test_dataloader)))
        test_loss,test_predict,test_labels,test_event_sentences_weight,test_score,test_precision,test_recall,test_f1 = eval_net(model,test_dataloader,test_sampler,CRLoss,show=True)

        accuracy_list.append(test_score)
        precision_list.append(test_precision)
        recall_list.append(test_recall)
        f1_list.append(test_f1)
        if test_score>best_predict:
            best_predict = test_score
            pred_result = pd.DataFrame(test_predict,columns=['pred'])
            pred_result['label'] = test_labels
            pred_result.to_csv('/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/pred_best.csv',index=False,encoding='utf-8')
            save_pkl(config.event_file,test_event_sentences_weight)
    accuracy_list = np.array(accuracy_list)
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    print('The mean and std of accuracy : ',accuracy_list.mean(),accuracy_list.std())
    print('The mean and std of precision : ',precision_list.mean(),precision_list.std())
    print('The mean and std of recall : ',recall_list.mean(),recall_list.std())
    print('The mean and std of f1 : ',f1_list.mean(),f1_list.std())
    print('The parameters are : ',config.mg_in_dim,
                            config.mg_out_dim,
                            config.sg_in_dim,
                            config.sg_out_dim,
                            config.na_in_dim,
                            config.na_out_dim,
                            config.ev_in_dim,
                            config.ev_hid_dim,
                            config.ev_out_dim,
                            config.en_in_dim,
                            config.en_out_dim,
                            config.num_layers,
                            config.num_heads)
    logger.info('result: \n')
    logger.info('The averge accuracy {:.6f}'.format(accuracy_list.mean()))
    logger.info('The averge precision {:.6f}'.format(precision_list.mean()))
    logger.info('The averge recall {:.6f}'.format(recall_list.mean()))
    logger.info('The averge f1score {:.6f}'.format(f1_list.mean()))

    
