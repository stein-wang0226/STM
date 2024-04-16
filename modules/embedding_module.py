import torch
from torch import nn
import numpy as np
import math
import csv
from model.temporal_attention import TemporalAttentionLayer
from model.HopAgg import HopAggLayer
from model.HopAgg import HopAggLayer2


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout, k):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.k = k

    def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        pass


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, k=2):
        super(GraphEmbedding, self).__init__(node_features, edge_features,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout, k)
        self.dlinear = nn.Linear(self.node_features.shape[1] * 2, self.node_features.shape[1])
        self.device = device
        self.n_neighbors = 0

    def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True, storage=None):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]
        # # todo 这里从node_featues修改为new_node_features

        if n_layers == 0:
            return source_node_features
        else:

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(neighbors,  # todo  递归
                                                         np.repeat(timestamps, n_neighbors),  #
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            # neighbor_embeddings = self.dlinear((torch.cat(neighbor_embeddings-source_node_features, neighbor_embeddings, dim=1)))
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            # target = 10
            # temp_noise = np.random.normal(loc=0, scale=2, size=172)
            # noise = target / np.linalg.norm(temp_noise) * temp_noise
            # edge_features += torch.tensor(noise).cuda()

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_features, source_nodes_time_embedding, # todo
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

            # todo 这里加了一层transformer
            # source_embedding = self.transformerLayer(source_embedding)
            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return None


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(  # todo transformer
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features
        )
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding


class GraghHopTransformer(GraphEmbedding):
    def __init__(self, node_features, edge_features, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, k=2, use_att=0, n_neighbors=20, type_of_find_k_closest="deascending"):
        super(GraghHopTransformer, self).__init__(node_features, edge_features,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout, k)
        self.type_of_find_k_closest = type_of_find_k_closest
        self.n_neighbors = n_neighbors
        self.attention_models = torch.nn.ModuleList([HopAggLayer2(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features, use_att=use_att, k=k, n_neighbors=n_neighbors
        )
            for _ in range(n_layers)])
        self.i = 1

    def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if n_layers == 0:
            return source_node_features
        else:
            neighbors, edge_idxs, edge_times = self.neighbor_finder.find_k_closest_byTimeAndSpace(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors, K=self.k,
                type_of_find_k_closest=self.type_of_find_k_closest
            )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

            edge_deltas = timestamps[:, np.newaxis, np.newaxis] - edge_times

            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(neighbors,  # todo
                                                         np.repeat(timestamps, self.k * n_neighbors),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            # todo 这里修改了一下，加了一个2，表明2条邻居
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), self.k, effective_n_neighbors, -1)

            # todo 修改成了flatten()
            edge_time_embeddings = self.time_encoder(torch.unsqueeze(edge_deltas_torch.flatten(), dim=1))
            # edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_features,  # todo聚合
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask, n_neighbors, self.k)

            # 这里加了一层transformer
            # source_embedding = self.transformerLayer(source_embedding)
            return source_embedding

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask, n_neighbors, k):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding = attention_model(source_node_features,  #
                                           source_nodes_time_embedding,
                                           neighbor_embeddings,
                                           edge_time_embeddings,
                                           edge_features,
                                           mask, n_neighbors, k)

        return source_embedding


def get_embedding_module(module_type, node_features, edge_features, neighbor_finder,  #
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=20, k=2, use_att=0, type_of_find_k_closest="deascending"):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout)
    elif module_type == "HopTransformer":
        return GraghHopTransformer(node_features=node_features,
                                   edge_features=edge_features,
                                   neighbor_finder=neighbor_finder,
                                   time_encoder=time_encoder,
                                   n_layers=n_layers,
                                   n_node_features=n_node_features,
                                   n_edge_features=n_edge_features,
                                   n_time_features=n_time_features,
                                   embedding_dimension=embedding_dimension,
                                   device=device,
                                   n_heads=n_heads, dropout=dropout, k=k
                                   , use_att=use_att, n_neighbors=n_neighbors,
                                   type_of_find_k_closest=type_of_find_k_closest)

    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
