import logging
import numpy as np
import torch
import argparse
from utils.utils import MergeLayer, get_previous_timestamp
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from modules.timeline import TimeAggModel


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, embedding_module_type="graph_attention", n_neighbors=20,
                 use_time_line=False,
                 time_line_length=1, agg_time_line="auto-encoder",
                 k=2,
                 data="DGraph",use_att = 0,
                 type_of_find_k_closest="descending"):
        super(TGN, self).__init__()
        self.use_att = use_att
        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.agg_time_line = agg_time_line
        self.data = data
        self.type_of_find_k_closest = type_of_find_k_closest

        if data == "DGraph":
            self.node_raw_features = node_features.to(device)
            self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        else:
            #  todo 这里采用新的数据集进行了修改
            self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
            self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type

        self.time_encoder = TimeEncode(dimension=self.n_node_features)

        self.use_time_line = use_time_line
        self.time_line_length = time_line_length

        self.k = k

        if self.use_time_line:
            self.agg_time_line = TimeAggModel(  # todo 帧间聚合
                time_line_length=self.time_line_length,
                feat_dim=self.embedding_dimension,
                agg_time_line=agg_time_line,  # todo
                # linear、auto-encoder
            )

        self.embedding_module_type = embedding_module_type

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     n_neighbors=self.n_neighbors,
                                                     k=self.k, use_att=self.use_att,
                                                     type_of_find_k_closest=self.type_of_find_k_closest
                                                     )

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

        self.compute_pos_by_src_layer = torch.nn.Linear(self.n_node_features, 1)  # todo

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])

        memory = None
        time_diffs = None

        # Compute the embeddings using the embedding module
        if self.use_time_line:
            # 这一行是得到一个节点的所有之前的时间
            prev_src_timestamps = self.neighbor_finder.get_time_stamps(source_nodes, edge_times)
            prev_dst_timestamps = self.neighbor_finder.get_time_stamps(destination_nodes, edge_times)
            # select time stamps to form a time line

            for i in range(self.time_line_length):
                selected_src_timestamps = []
                selected_dst_timestamps = []

                for src_timestamps, dst_timestamps in \
                        zip(prev_src_timestamps, prev_dst_timestamps):
                    # 按照个数来，for循环的所有个数为src_nodes的个数
                    selected_src_timestamps.append(get_previous_timestamp(src_timestamps, i, n_neighbors))
                    selected_dst_timestamps.append(get_previous_timestamp(dst_timestamps, i, n_neighbors))

                nodes = np.concatenate([nodes, source_nodes, destination_nodes])
                timestamps = np.concatenate(
                    [timestamps, selected_src_timestamps, selected_dst_timestamps]
                )

        node_embedding = self.embedding_module.compute_embedding(source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        # n_samples表示src_nodes的总结点数
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        if self.use_time_line:
            prev_node_embeddings = []
            for time_line_idx in range(self.time_line_length):
                prev_src_node_embedding = node_embedding[2 * n_samples + 2 * n_samples * time_line_idx:
                                                         3 * n_samples + 2 * n_samples * time_line_idx]
                prev_dst_node_embedding = node_embedding[3 * n_samples + 2 * n_samples * time_line_idx:
                                                         4 * n_samples + 2 * n_samples * time_line_idx]
                prev_node_embeddings.append(torch.cat([prev_src_node_embedding,
                                                       prev_dst_node_embedding]))
            cur_local_embed = torch.cat([source_node_embedding,
                                         destination_node_embedding])
            embeddings_after_time_agg = self.agg_time_line(cur_local_embed=cur_local_embed,
                                                           prev_local_embeds=prev_node_embeddings) # 200 2 17   , 200 2 17
            source_node_embedding = embeddings_after_time_agg[:n_samples]
            destination_node_embedding = embeddings_after_time_agg[n_samples: 2 * n_samples]

        return source_node_embedding, destination_node_embedding

    def compute_probabilities(self, source_nodes, destination_nodes, edge_times,
                              edge_idxs, n_neighbors=20):
        n_samples = len(source_nodes)
        # todo 计算嵌入
        source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors)

        if self.data == "DGraph":
            source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings(
                source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors)
            probOfSrc = self.compute_pos_by_src_layer(source_node_embedding)
            probOfDes = self.compute_pos_by_src_layer(destination_node_embedding)
        else:
            score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                        torch.cat([destination_node_embedding,
                                                   destination_node_embedding])).squeeze(dim=0)
            probOfSrc = score[:n_samples]
            probOfDes = score[n_samples:]
        return probOfSrc.sigmoid(), probOfDes.sigmoid()

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
