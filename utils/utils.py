import numpy as np
import torch


def get_previous_timestamp(timestamps, i, n_neighbors):
    # i表示第几个time_line，即第几个搜集出来的图
    # n_neighbors就表示frame_length
    target = int(((i + 1) * n_neighbors) / 2)
    # 如果小于这个时间的所有节点数量大于目标数目，那么取最近的倒数第target个,其实这里就是在进行分帧了
    if len(timestamps) >= target:
        return timestamps[-target]
    else:
        return 0


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class MLP2(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 2)
        self.fc_3 = torch.nn.Linear(2, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)

    def draw(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        return self.sig(self.act(self.fc_2(x)))


def special_optimize(name, a, b):
    if name == 'mooc':
        return b, a
    else:
        return a,b

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        # todo: here may get wrong negative neighbor
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, max_node_idx=None, sample_mode='random', hard_sample=False):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, sample_mode=sample_mode, hard_sample=hard_sample)


class NeighborFinder:
    def __init__(self, adj_list, seed=None, sample_mode='random', hard_sample=False):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # todo We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            # 将元组分开，添加到邻居列表中
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        self.sample_mode = sample_mode
        self.hard_sample = hard_sample

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], \
               self.node_to_edge_timestamps[src_idx][:i]

    def get_time_stamps(self, src_idx_l, cur_time_l):
        batch_time_stamps = []
        for src_idx, cur_time in zip(src_idx_l, cur_time_l):
            # print(src_idx, cur_time)
            # 得到一个点所有的生命周期，即出现和其他点交互的所有时间
            total_time_stamps = self.node_to_edge_timestamps[src_idx]
            # print(len(total_time_stamps))
            pos = 0
            # 找到在当前时间之前的所有时间节点
            for i in total_time_stamps:
                if i >= cur_time:
                    break
                pos += 1
            batch_time_stamps.append(total_time_stamps[: pos])
        return batch_time_stamps

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)
            # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.sample_mode == 'random':
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                elif self.sample_mode == 'time':
                    if self.hard_sample:
                        # Take most recent interactions
                        source_edge_times = source_edge_times[-n_neighbors:]
                        source_neighbors = source_neighbors[-n_neighbors:]
                        source_edge_idxs = source_edge_idxs[-n_neighbors:]

                        assert (len(source_neighbors) <= n_neighbors)
                        assert (len(source_edge_times) <= n_neighbors)
                        assert (len(source_edge_idxs) <= n_neighbors)

                        neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                        edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                        edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
                    else:
                        if len(source_neighbors) > 20 * n_neighbors:
                            source_neighbors = source_neighbors[-20 * n_neighbors:]
                            source_edge_times = source_edge_times[-20 * n_neighbors:]
                            source_edge_idxs = source_edge_idxs[-20 * n_neighbors:]

                        sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
                        neighbors[i, :] = source_neighbors[sampled_idx]
                        edge_times[i, :] = source_edge_times[sampled_idx]
                        edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                        pos = edge_times[i, :].argsort()
                        neighbors[i, :] = neighbors[i, :][pos]
                        edge_times[i, :] = edge_times[i, :][pos]
                        edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    exit('wrong sample mode')

        return neighbors, edge_idxs, edge_times

    def find_k_hop(self, source_nodes, timestamps, n_neighbors=20, K=2):
        # assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1

        neighbors = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)
        edge_times = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)
        edge_idxs = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)

        query_nodes = source_nodes

        neighbors_hop1, edge_idxs_hop1, edge_times_hop1 = self.get_temporal_neighbor(query_nodes, timestamps,
                                                                                     n_neighbors)
        neighbors[:, 0, :] = neighbors_hop1
        edge_idxs[:, 0, :] = edge_idxs_hop1
        edge_times[:, 0, :] = edge_times_hop1

        neighbors_time_list = []
        for neighbor_arr, timestamp in zip(neighbors_hop1, timestamps):
            length = len(neighbor_arr)
            neighbors_time = [timestamp] * length
            neighbors_time_list.append(np.array(neighbors_time))

        for k in range(1, K, 1):
            for i, (neighbor_arr, neighbors_time) in enumerate(zip(neighbors_hop1, neighbors_time_list)):
                nonzero_indices = np.nonzero(neighbor_arr)
                nonzero_elements = neighbor_arr[nonzero_indices]

                nonzero_time = neighbors_time[nonzero_indices]
                if len(nonzero_elements) != 0:
                    neighbors_hop2, edge_idxs_hop2, edge_times_hop2 = self.get_temporal_neighbor(nonzero_elements,
                                                                                                 nonzero_time,
                                                                                                 n_neighbors)
                    # 堆叠一下，用于比较得出不重复元素
                    stack = np.dstack((neighbors_hop2, edge_times_hop2, edge_idxs_hop2))
                    # 将元组转换为类型
                    void_arr = stack.view(np.dtype((np.void, stack.dtype.itemsize * stack.shape[2])))

                    # 获取不重复的元组
                    unique_void = np.unique(void_arr)

                    # 将void类型转换回元组
                    unique_tuples = unique_void.view(stack.dtype).reshape(-1, stack.shape[2])

                    # 根据时间进行对unique_tuples排序
                    unique_tuples_sorted = unique_tuples[unique_tuples[:, 1].argsort()]

                    neighbors[i, k, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 0]
                    edge_idxs[i, k, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 2]
                    edge_times[i, k, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 1]

            neighbors_hop1 = neighbors[:, k, :]
        return neighbors, edge_idxs, edge_times

    def get_temporal_hopK_neighbors(self, source_nodes, timestamps, n_neighbors=20, K=2):
        # assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1

        neighbors = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)
        edge_times = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)
        edge_idxs = np.zeros((len(source_nodes), K, tmp_n_neighbors)).astype(np.int32)

        query_nodes = source_nodes

        neighbors_hop1, edge_idxs_hop1, edge_times_hop1 = self.get_temporal_neighbor(query_nodes, timestamps,
                                                                                     n_neighbors)
        neighbors[:, 0, :] = neighbors_hop1
        edge_idxs[:, 0, :] = edge_idxs_hop1
        edge_times[:, 0, :] = edge_times_hop1

        neighbors_time_list = []
        for neighbor_arr, timestamp in zip(neighbors_hop1, timestamps):
            length = len(neighbor_arr)
            neighbors_time = [timestamp] * length
            neighbors_time_list.append(np.array(neighbors_time))

        #
        for i, (neighbor_arr, neighbors_time) in enumerate(zip(neighbors_hop1, neighbors_time_list)):
            nonzero_indices = np.nonzero(neighbor_arr)
            nonzero_elements = neighbor_arr[nonzero_indices]

            nonzero_time = neighbors_time[nonzero_indices]
            if len(nonzero_elements) != 0:
                neighbors_hop2, edge_idxs_hop2, edge_times_hop2 = self.get_temporal_neighbor(nonzero_elements,
                                                                                             nonzero_time,
                                                                                             n_neighbors)
                # 堆叠一下，用于比较得出不重复元素
                stack = np.dstack((neighbors_hop2, edge_times_hop2, edge_idxs_hop2))
                # 将元组转换为类型
                void_arr = stack.view(np.dtype((np.void, stack.dtype.itemsize * stack.shape[2])))

                # 获取不重复的元组
                unique_void = np.unique(void_arr)
                # 将void类型转换回元组
                unique_tuples = unique_void.view(stack.dtype).reshape(-1, stack.shape[2])
                # todo 根据时间进行对unique_tuples排序 取前20
                unique_tuples_sorted = unique_tuples[unique_tuples[:, 1].argsort()]

                neighbors[i, 1, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 0]
                edge_idxs[i, 1, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 2]
                edge_times[i, 1, -len(unique_tuples_sorted):] = unique_tuples_sorted[:n_neighbors, 1]

        return neighbors, edge_idxs, edge_times
