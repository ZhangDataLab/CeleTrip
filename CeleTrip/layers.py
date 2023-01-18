import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling
####################################################
import torch.nn as nn
import torch
import logging
from scipy.stats import t
import math

def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
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
    # 此处的k是一个向量，每一维是每个子图保留的节点数。k可以设置一个下限，比如为滑动窗口的大小
    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k

class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):
        # compute socres
        score = self.score_layer(graph, feature).squeeze()
        # get idx, Z_mask = Z_idx
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        # X_out = X'*Z_mask (X'=X_idx)
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.8):
        super(ConvPoolBlock, self).__init__()
        self.conv = GraphConv(in_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()   
    
    def forward(self, graph, feature):
        out = F.relu(self.conv(graph, feature))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, g_out

class OrientedPool(torch.nn.Module):
    """
    OrientedPool is modified based on SAGPool.
    The Self-Attention Pooling layer in paper(SAGPool)
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(OrientedPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity
        # 相似度度量
        self.cos_score = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.att_linear = nn.Linear(2, 1, bias=True)
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor, need_guide = False, special_nodes_idx = None):
        score = self.score_layer(graph, feature).squeeze()
        
        # 可以在这里增加与对应节点相似度的计算
        if need_guide: # 需要用关键信息进行引导
            # unbatch
            # sim score
            # 计算每个图上每个节点对应的相似度得分，然后存在每个图的 sim_score 中，按图进行更新
            graph.ndata["sim_score"] = torch.ones_like(score)
            graph.ndata["feat"] = feature
            graph_list = dgl.unbatch(graph)
            # for each_graph, person_idx, loc_idx in zip(graph_list, special_node_index):
            #     # update graph feature
            #     pass
            for each_graph, nodes_idx in zip(graph_list, special_nodes_idx):
                 for node_idx in nodes_idx:
                    if node_idx == -1: # 不存在special node，则不更新相似度得分
                         continue
                     # get special feature
                    node_feature = each_graph.ndata["feat"][node_idx]
                     # repeat it to martix
                    node_feature = node_feature.repeat(each_graph.num_nodes(), 1)
                    cos_s = self.cos_score(node_feature, each_graph.ndata["feat"])
                     # update score [累乘相似度]
                    each_graph.ndata["sim_score"] = each_graph.ndata["sim_score"] * cos_s
            # batch
            graph = dgl.batch(graph_list)
            sim_score = graph.ndata["sim_score"]
            # additive attention
            score = torch.reshape(score, (score.shape[0], 1))
            sim_score = torch.reshape(sim_score, (sim_score.shape[0], 1))
            con_score = torch.cat([score, sim_score], 1)
            score = torch.flatten(torch.tanh(self.att_linear(con_score)))
            
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm