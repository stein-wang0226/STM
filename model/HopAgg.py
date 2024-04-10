import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HopAggLayer(nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1, use_att=0, k=1, n_neighbors=20):
        super(HopAggLayer, self).__init__()
        self.use_att = use_att
        self.n_head = n_head

        self.node_feat_dim = n_node_features
        self.edge_feat_dim = n_edge_features
        self.time_dim = time_dim

        # 设置这一层的意义是将邻居特征和时间特征和边的特征结合起来， 然后重新得到一个向量
        self.neighborTimeCatLinear = nn.Linear(self.node_feat_dim + self.time_dim + self.edge_feat_dim,
                                               self.node_feat_dim)
        # 设置这一层的意义是将原始节点的特征和时间特征、边的特征结合起来，得到一个向量
        self.scrTimeCatLinear = nn.Linear(self.node_feat_dim + self.time_dim, self.node_feat_dim)
        # 设置这一层的目的就是采用NagTransformer那篇文章，进行注意力的计算
        self.att_layer = nn.Linear(2 * self.node_feat_dim, 1)
        # todo 设置这些层的目的是采取相似性增强
        self.hoLayer = nn.Linear(2 * self.node_feat_dim, 1)
        self.heLayer = nn.Linear(2 * self.node_feat_dim, 1)
        self.catLayer = nn.Linear(n_neighbors * self.node_feat_dim, self.node_feat_dim)
        self.catLayer2 = nn.Linear(2 * self.node_feat_dim, self.node_feat_dim)
        self.catLayer3 = nn.Linear(4 * self.node_feat_dim, self.node_feat_dim)
        self.catLayer4 = nn.Linear(5 * self.node_feat_dim, self.node_feat_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features, neighbors_padding_mask,
                n_neighbors=20, k=2):
        # 将他们都扩展成二维向量,使用TGAT的加入时间、边、点的特征，最终的维度是（总邻居个数，feat_dim）
        neighbors_features = neighbors_features.view(-1, self.node_feat_dim)
        neighbors_time_features = neighbors_time_features.view(-1, self.time_dim)
        edge_features = edge_features.view(-1, self.edge_feat_dim)
        neighbors_padding_mask = neighbors_padding_mask.flatten()
        new_neighbors_features = self.neighborTimeCatLinear(
            torch.cat((neighbors_features, neighbors_time_features, edge_features), dim=1)) \
                                 + neighbors_features  # todo
        # 将padding_mask中的True值替换成0，对于对于邻居位置标志为0的进行掩码
        new_neighbors_features[neighbors_padding_mask, :] = 0

        # 将src_time_features去掉一个维度，并综合时间特征，最终得到的维度是(源节点个数，feat_dim)
        src_time_features = src_time_features.squeeze()
        new_src_node_features = self.scrTimeCatLinear(
            torch.cat((src_node_features, src_time_features), dim=1))  # 600  172
        # K表示K跳邻居，n_neighbors表示n_neighbors个邻居，最终得到的维度是(源节点个数，k条邻居，每条邻居的个数，feat_dim)
        new_neighbors_features = new_neighbors_features.view(len(src_node_features), k, n_neighbors,
                                                             -1)  # 12000 172--> 600 1 20 172
        # 将邻居特征中的每跳邻居的特征综合起来称为1维，然后直接消除这个维度 600 1 20单跳邻居 172    600 1 172
        # todo 搞个参数 cat_num表示拼接长度？

        # todo 差异聚合
        # new_neighbors_features = new_neighbors_features.view(len(new_src_node_features),k*n_neighbors, -1)
        # # 执行减法操作
        # diff_aggregation = new_neighbors_features - new_src_node_features.unsqueeze(1) # 600 ,20k,172  - 600,1,172
        # # 恢复张量形状
        # new_neighbors_features = diff_aggregation.view(len(new_src_node_features), k, n_neighbors, -1)
        # todo 1 sum 20
        # new_neighbors_features = torch.sum(new_neighbors_features, dim=2)  # todo1 init change into cat ?

        # todo 2 cat 20
        # new_neighbors_features = self.catLayer(
        #     new_neighbors_features.view(len(src_node_features), k, -1))  # change into cat 600 1 172
        # todo 3 sum10(cat 2)
        # (2)*10   600 1 20 172   --> 600 1 10 (2*172 )  /  600 1 5 (4*172 )
        # new_neighbors_features = torch.sum(self.catLayer2(
        #     new_neighbors_features.view(len(src_node_features), k, 10, -1)), dim=2)
        # todo 4 sum5(cat 4)
        # (2)*10   600 1 20 172   --> 600 1 10 (2*172 )  /  600 1 5 (4*172 )
        new_neighbors_features = torch.sum(self.catLayer3(
            new_neighbors_features.view(len(src_node_features), k,5,-1)),dim =2)
        # todo 5 sum4(cat 5)
        # (2)*10   600 1 20 172   --> 600 1 10 (2*172 )  /  600 1 4 (5*172 )
        # new_neighbors_features = torch.sum(self.catLayer4(
        #     new_neighbors_features.view(len(src_node_features), k, 4, -1)), dim=2)
        # todo 尝试差异聚合位置1

        new_src_node_features = new_src_node_features.unsqueeze(dim=1)  # todo 600 1 1 172
        new_src_node_features_repeat = new_src_node_features.repeat(1, k, 1)  # 600 1 20 172
        # 600 1 172()
        # todo 尝试差异聚合位置2
        # new_src_node_features_repeat = new_src_node_features_repeat - new_neighbors_features

        # todo 尝试相似性聚合
        # ho = self.hoLayer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
        # he = self.heLayer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
        # new_src_node_features_repeat = new_src_node_features_repeat + (ho-he)*new_neighbors_features


        # 计算注意力 todo 研究
        # att = self.att_layer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
        # print(new_neighbors_features.shape,att.shape) # todo 600,1,172特征维度   |  600 1 1
        # neighbor_tensor = new_neighbors_features*att

        neighbor_tensor = new_neighbors_features
        # 600, 1, 172
        # 将多跳邻居 的特征全部加起来, 最终维度变为(源节点个数，feat_dim) # todo     change into cat ?
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        # 600, k, 172

        if self.use_att:
            # todo 计算跳之间注意力权重2
            attn_weights = self.att_layer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
            attn_weights = F.softmax(attn_weights, dim=1)
            # 利用注意力权重对邻居特征进行加权求和
            attended_neighbor_features = (attn_weights * new_neighbors_features)
            # 使用残差连接并压缩维度
            output = (new_src_node_features + attended_neighbor_features).squeeze()
        # todo
        else:
            # 使用残差连接 600 172
            output = (new_src_node_features + neighbor_tensor).squeeze()

        # output = self.dropout(output)

        return output


class HopAggLayer2(nn.Module):  # todo  其他数据集测试
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1, use_att=0, k=1, n_neighbors=20):
        super(HopAggLayer2, self).__init__()

        self.use_att = use_att
        self.n_head = n_head

        self.node_feat_dim = n_node_features
        self.edge_feat_dim = n_edge_features
        self.time_dim = time_dim

        # 设置这一层的意义是将邻居特征和时间特征和边的特征结合起来， 然后重新得到一个向量
        self.neighborTimeCatLinear = nn.Linear(self.node_feat_dim + self.time_dim + self.edge_feat_dim,
                                               self.node_feat_dim)
        # 设置这一层的意义是将原始节点的特征和时间特征、边的特征结合起来，得到一个向量
        self.scrTimeCatLinear = nn.Linear(self.node_feat_dim + self.time_dim, self.node_feat_dim)
        # 设置这一层的目的就是采用NagTransformer那篇文章，进行注意力的计算
        self.att_layer = nn.Linear(2 * self.node_feat_dim, 1)
        # todo 设置这些层的目的是采取相似性增强
        self.hoLayer = nn.Linear(2 * self.node_feat_dim, 1)
        self.heLayer = nn.Linear(2 * self.node_feat_dim, 1)
        self.catLayer = nn.Linear(n_neighbors * self.node_feat_dim, self.node_feat_dim)
        self.catLayer2 = nn.Linear(2 * self.node_feat_dim, self.node_feat_dim)
        self.catLayer3 = nn.Linear(4 * self.node_feat_dim, self.node_feat_dim)

    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features, neighbors_padding_mask,
                n_neighbors=10, k=2):
        # 将他们都扩展成二维向量,使用TGAT的加入时间、边、点的特征，最终的维度是（总邻居个数，feat_dim）
        neighbors_features = neighbors_features.view(-1, self.node_feat_dim)
        neighbors_time_features = neighbors_time_features.view(-1, self.time_dim)
        edge_features = edge_features.view(-1, self.edge_feat_dim)
        neighbors_padding_mask = neighbors_padding_mask.flatten()
        new_neighbors_features = self.neighborTimeCatLinear(
            torch.cat((neighbors_features, neighbors_time_features, edge_features), dim=1)) \
                                 + neighbors_features  # todo
        # 将padding_mask中的True值替换成0，对于对于邻居位置标志为0的进行掩码
        new_neighbors_features[neighbors_padding_mask, :] = 0

        # 将src_time_features去掉一个维度，并综合时间特征，最终得到的维度是(源节点个数，feat_dim)
        src_time_features = src_time_features.squeeze()
        new_src_node_features = self.scrTimeCatLinear(torch.cat((src_node_features, src_time_features), dim=1))
        # K表示K跳邻居，n_neighbors表示n_neighbors个邻居，最终得到的维度是(源节点个数，k条邻居，每条邻居的个数，feat_dim)
        new_neighbors_features = new_neighbors_features.view(len(src_node_features), k, n_neighbors,
                                                             -1)  # 12000 172--> 600 1 20 172
        # 将邻居特征中的每跳邻居的特征综合起来称为1维，然后直接消除这个维度 600 1 20单跳邻居 172  600 1 172
        # todo 搞个参数 cat_num表示拼接长度？
        # todo 1 sum 20
        new_neighbors_features = torch.sum(new_neighbors_features, dim=2)  # todo init change into cat ?
        # todo 2 cat 20
        # new_neighbors_features = self.catLayer(
        #     new_neighbors_features.view(len(src_node_features), k, -1))  # todo init change into cat 600 1 172
        # todo 3 sum10(cat 2)
        # (2)*10   600 1 20 172   --> 600 1 10 (2*172 )  /  600 1 5 (4*172 )
        # new_neighbors_features = torch.sum(self.catLayer2(
        #     new_neighbors_features.view(len(src_node_features), k,10,-1)),dim =2)
        # todo 4 sum5(cat 4)
        # (2)*10   600 1 20 172   --> 600 1 10 (2*172 )  /  600 1 5 (4*172 )
        # new_neighbors_features = torch.sum(self.catLayer3(
        #     new_neighbors_features.view(len(src_node_features), k,5,-1)),dim =2)

        new_src_node_features = new_src_node_features.unsqueeze(dim=1)
        new_src_node_features_repeat = new_src_node_features.repeat(1, k, 1)
        # 600 1 172()

        # neighbor_tensor = new_neighbors_features
        # # 600, 1, 172
        # # 将多跳邻居 的特征全部加起来, 最终维度变为(源节点个数，feat_dim) # todo     change into cat ?
        # neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        # 600, k, 172

        # if self.use_att:
        #     # todo 计算跳之间注意力权重2
        #     attn_weights = self.att_layer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
        #     attn_weights = F.softmax(attn_weights, dim=1)
        #     # 利用注意力权重对邻居特征进行加权求和
        #     attended_neighbor_features = (attn_weights * new_neighbors_features)
        #
        #     # 使用残差连接并压缩维度
        #     neighbor_tensor = torch.sum(attended_neighbor_features, dum=1, keepdim=True)
        #
        #     output = (new_src_node_features + neighbor_tensor).squeeze()  # todo
        # else:
        #     # 使用残差连接 600 172
        #     neighbor_tensor = new_neighbors_features
        #     # 600, 1, 172
        #     # 将多跳邻居 的特征全部加起来, 最终维度变为(源节点个数，feat_dim) # todo     change into cat ?
        #     neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        #     output = (new_src_node_features + neighbor_tensor).squeeze()
        #
        # return output
        if self.use_att:
            att = self.att_layer(torch.cat((new_src_node_features_repeat, new_neighbors_features), dim=2))
            att = F.softmax(att, dim=1)
            neighbor_tensor = att * new_neighbors_features

        else:
            neighbor_tensor = new_neighbors_features

        # 将多跳邻居的特征全部加起来, 最终维度变为(源节点个数，feat_dim)
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        # 使用残差连接
        output = (new_src_node_features + neighbor_tensor).squeeze()
        # todo 直接拼接起来
        # neighbor_tensor = self.catLayer2(neighbor_tensor.view(len(src_node_features), -1))
        # output = (new_src_node_features.squeeze()+neighbor_tensor)
        return output
