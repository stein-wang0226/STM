import numpy as np
import random
import pandas as pd
import torch_geometric as tg
import torch
from utils.utils import special_optimize
import networkx as nx
from tqdm import trange
from scipy.stats import pearsonr

from option import args

import logging

import torch

from pathlib import Path
from datetime import datetime


### set up logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
log_filename = 'log/{}-{}.log'.format(
    args.prefix,
    '--'.join([args.data, args.embedding_module, f'k={args.k}', f'time_line_length={args.time_line_length}',
               f'node_fetch={args.node_fetch}', f'use_att={args.use_att}', f'n_degree={args.n_degree}', f'bs={args.bs}',
               f'ls={args.lr}', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')])
)
fh = logging.FileHandler(log_filename)

fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, entropy=None, labelsOfDes=None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.entropy = entropy
        self.labelsOfDes = labelsOfDes


def get_data_node_classification(dataset_name, use_validation=False, path_prefix='./',
                                 attack=False,
                                 attack_range=None, shift=False, shift_size=None,
                                 attack_edge=False, attack_edge_percent=None, attack_version=1
                                 ):
    ### Load data and train val test split
    if attack_edge:
        print('use edge attack')
        edge_features = np.load(path_prefix + 'ml_{}_edge_attack{}.npy'.format(dataset_name, attack_edge_percent))
        graph_df = pd.read_csv(path_prefix + 'ml_{}_edge_attack{}.csv'.format(dataset_name, attack_edge_percent))
        node_features = np.load(path_prefix + 'ml_{}_node.npy'.format(dataset_name))
    else:
        if attack:
            if shift:
                print('use shift attack')
                assert shift_size is not None
                edge_features = np.load(path_prefix + 'ml_{}_attack_shift_{}.npy'.format(dataset_name, shift_size))
            else:
                print('use normal attack')
                assert attack_range is not None
                edge_features = np.load(
                    path_prefix + 'ml_{}_attack_{}_{}.npy'.format(dataset_name, attack_range, attack_version))
        else:
            edge_features = np.load(path_prefix + 'ml_{}.npy'.format(dataset_name))
        graph_df = pd.read_csv(path_prefix + 'ml_{}.csv'.format(dataset_name))
        node_features = np.load(path_prefix + 'ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, path_prefix='./utils/data',
             node_fetch=True):  # ./utils/data   --源数据集划分-->    ./utils/data/origin
    path_prefix = ".\\utils\\data"
    if dataset_name == 'DGraph':
        datapath = path_prefix + "/dgraphfin.npz"
        origin_data = np.load(datapath)
        data = build_tg_data(is_undirected=False, datapath=datapath)
        logger.info("读取数据完毕")
        return process_data(data, node_fetch=node_fetch)
    else:
        g_df = pd.read_csv(path_prefix + '/ml_{}.csv'.format(dataset_name))
        e_feat = np.load(path_prefix + '/ml_{}.npy'.format(dataset_name))
        n_feat = np.load(path_prefix + '/ml_{}.npy'.format(dataset_name))
        val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
        logger.info("读取数据完毕")

        src_l = g_df.u.values
        dst_l = g_df.i.values
        e_idx_l = g_df.idx.values
        label_l = g_df.label.values
        ts_l = g_df.ts.values

        train_mask = ts_l <= val_time
        test_mask = ts_l > test_time
        val_mask = np.logical_and(ts_l <= test_time, ts_l > val_time)
        # full_data
        full_data = Data(src_l, dst_l, ts_l, e_idx_l, label_l)
        # train_data

        train_data = Data(src_l[train_mask], dst_l[train_mask], ts_l[train_mask], e_idx_l[train_mask],
                          label_l[train_mask])
        # test_data
        test_data = Data(src_l[test_mask], dst_l[test_mask], ts_l[test_mask], e_idx_l[test_mask],
                         label_l[test_mask])
        # val_data
        val_data = Data(src_l[val_mask], dst_l[val_mask], ts_l[val_mask], e_idx_l[val_mask],
                        label_l[val_mask])
        val_data, test_data = special_optimize(dataset_name, val_data, test_data)
        return n_feat, e_feat, full_data, train_data, test_data, val_data


def get_data_diff(dataset_name, path_prefix='./', ):
    g_df = pd.read_csv(path_prefix + 'ml_{}.csv'.format(dataset_name))
    e_features = np.load(path_prefix + 'ml_{}.npy'.format(dataset_name))
    n_features = np.load(path_prefix + 'ml_{}_node.npy'.format(dataset_name))
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    valid_train_flag = (ts_l <= test_time)
    valid_val_flag = (ts_l <= test_time)
    assignment = np.random.randint(0, 10, len(valid_train_flag))
    valid_train_flag *= (assignment >= 2)
    valid_val_flag *= (assignment < 2)
    valid_test_flag = ts_l > test_time

    # 这里是没有使用tune
    valid_train_flag = (ts_l <= test_time)
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    # use the true test dataset
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

    # full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    full_data = Data(src_l, dst_l, ts_l, e_idx_l, label_l)
    train_data = Data(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l)
    val_data = Data(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l)
    test_data = Data(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l)
    # return node_features, edge_features, full_data, train_data, val_data, test_data, \
    #            new_node_val_data, new_node_test_data
    return n_features, e_features, full_data, train_data, val_data, test_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


def build_tg_data(is_undirected=True, datapath=None):
    origin_data = np.load(datapath)
    data = tg.data.Data()  # data (x = [3700550,17]  y = [3700550], edge_index = [2, 4300999])
    data.x = torch.tensor(origin_data['x']).float()
    data.y = torch.tensor(origin_data['y']).long()
    data.edge_index = torch.tensor(origin_data['edge_index']).long().T
    data.train_mask = torch.tensor(origin_data['train_mask']).long()
    data.val_mask = torch.tensor(origin_data['valid_mask']).long()
    data.test_mask = torch.tensor(origin_data['test_mask']).long()
    data.edge_time = torch.tensor(origin_data['edge_timestamp']).long()
    if (is_undirected):
        data.edge_index = tg.utils.to_undirected(data.edge_index)
    return data


from collections import deque


def bfs_within_d2(G, source, d2, labels):
    """
    使用 BFS 算法获取源节点 v 在距离不超过 d2 的范围内的bg节点
    """
    visited = set()
    queue = deque([(source, 0)])  # 将源节点放入队列，并记录距离为 0
    within_d2_nodes = set()  # 存储距离源节点不超过 d2 的范围内的节点
    while queue:
        node, distance = queue.popleft()

        if distance > d2:
            break
        if labels[node] in [2, 3]:  # bg
            within_d2_nodes.add(node)  # 将节点添加到结果集中
        visited.add(node)
        if distance < d2:
            for neighbor in G[node]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
    return list(within_d2_nodes)


from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

def process_data(data, max_time_steps=32, node_fetch=True):
    # 对边的时间进行处理  # todo Dgraph
    data.edge_time = data.edge_time - data.edge_time.min()  # process edge time
    data.edge_time = data.edge_time / data.edge_time.max()
    data.edge_time = (data.edge_time * max_time_steps).long()
    data.edge_time = data.edge_time.view(-1, 1).float()
    node_features = data.x
    labels = data.y

    # 先固定点, 从0开始1100000选择个点
    random.seed(123)
    random.seed(123)
    ori_len = len(labels)  # 即结点数
    length =2500000
    length = args.DGraph_size
    logger.info(f'Graph size:{length}')
    nodes_slices = np.arange(0, length)
    data.edge_index = data.edge_index.T  #
    elist = data.edge_index  # [n,2]
    mask = ~np.isin(elist, nodes_slices)  # [[1,0],[],[],[]]  [n,2]
    elist[mask] = -1  # 非区间内的-1   [[-1,xx][][][]
    elist = elist.cpu().numpy()
    data.edge_time = data.edge_time.cpu().numpy()
    rows_to_delete = np.where(np.any(elist == -1, axis=1))  # 含-1的边
    elist = np.delete(elist, rows_to_delete, axis=0)  # 删除
    timestamps = np.delete(data.edge_time, rows_to_delete, axis=0)
    timestamps = timestamps.squeeze()
    # nodes  = np.unique(elist.flatten()) # elist 中所有点
    # new_labels = torch.zeros(elist.shape[0])
    # edge_features = np.zeros((len(elist), 17))
    use_fetch = node_fetch  # bool
    if use_fetch:
        save_path = f'utils/fetched_data.npz'
        logger.info('start node fetching ')
        ######################################################### todo node-fetch
        saved = True
        if not saved or not os.path.exists(save_path):
            # todo 加入到args
            d1 = args.d1   # todo   sd小于的d1的保留   越小删越多？ 发现>=3 都相等(选不了的都是没有与两个target 连通的bg node ——通过隶属fetch)
            d2 = args.d2  # todo 取d2跳的k个最相似的
            k = args.k
            save_path = f'utils/fetched_data_len{length}_{d1}{d2}{k}.npz'
            nodes = np.unique(elist.flatten())  # elist 中所有点
            ############################# todo begin
            # 添加节点 构图
            G = nx.Graph()
            G.add_edges_from(elist.tolist())
            logger.info("原子图点数:{}，边数：{}".format(G.number_of_nodes(), G.number_of_edges()))
            density = nx.density(G)
            average_degree = sum(dict(G.degree()).values()) / len(G)
            logger.info(f"Density of the graph: {density}")
            logger.info(f"Average degree of the graph: {average_degree}")
            # 将列表 nodes 转换为集合
            # 使用集合来查找节点
            bg_nodes = [i for i in nodes if labels[i] in [2, 3]]
            target_nodes = [i for i in nodes if labels[i] in [0, 1]]
            save_bg_node_l = []
            logger.info(f'当前总节点数:{len(nodes),} ，背景节点数：{len(bg_nodes)},目标节点数：{len(target_nodes)}')
            # todo step1 bfs Bridging Background Node Fetching 背景节点j 到两最近目标节点距离之和<d1
            for node_i in trange(len(bg_nodes)):
                bg_node = bg_nodes[node_i]
                shortest_paths = nx.single_source_shortest_path_length(G, source=bg_node)  # todo 返回最短路字典
                # 删除背景节点的最短路径信息
                del shortest_paths[bg_node]  #
                # 按最短路径长度排序
                sorted_shortest_paths = sorted(shortest_paths.items(), key=lambda x: x[1])
                # 获取最短和第二短距离
                shortest_distances = [dist for node, dist in sorted_shortest_paths]

                # 获取最短和第二短距离
                closest_distance = shortest_distances[0]  # 最短距离
                second_closest_distance = shortest_distances[1] if len(shortest_distances) > 1 else float(
                    'inf')  # 第二短距离（如果存在）
                if second_closest_distance + closest_distance <= d1:
                    save_bg_node_l.append(bg_node)
            save_bg_node_l = list(np.unique(save_bg_node_l))
            del_cnt1 = len(bg_nodes) - len(save_bg_node_l)
            logger.info('Bridging Background Node Fetching完成')
            logger.info(f'当前背景节点数：{len(save_bg_node_l)},目标节点数：{len(target_nodes)}')
            # todo step2  bfs Affiliation Background Node Fetching. 在与target距离小于d2的背景结点中选择K个相关性最大的
            for node_i in trange(len(target_nodes)):
                target_node = target_nodes[node_i]
                bg_nodes_within_d2 = bfs_within_d2(G, target_node, d2, labels)

                # 计算与源节点的相关性，并选择相关性最大的 k 个节点
                bg_pcc_dict = {}  # 存储背景节点与源节点的相关性
                if len(bg_nodes_within_d2) < k:
                    save_bg_node_l += bg_nodes_within_d2  # todo bg全加
                    continue
                for bg_node in bg_nodes_within_d2:
                    feature1 = node_features[target_node]
                    feature2 = node_features[bg_node]
                    correlation, p_value = pearsonr(feature1, feature2)
                    bg_pcc_dict[bg_node] = correlation
                # 按相关性排序
                sorted_bg_pcc = sorted(bg_pcc_dict.items(), key=lambda item: item[1], reverse=True)
                # 选择与源节点相关性最大的 k 个节点
                if len(sorted_bg_pcc) < k:
                    selected_bg_nodes = [item[0] for item in sorted_bg_pcc]
                else:
                    selected_bg_nodes = [item[0] for item in sorted_bg_pcc[:k]]
                    # 将选定的节点加入结果列表
                save_bg_node_l.extend(selected_bg_nodes)  # todo

            save_bg_node_l = list(np.unique(save_bg_node_l))
            save_node_l = np.unique(save_bg_node_l + target_nodes)
            del_mask = ~np.isin(elist, save_node_l)
            # # todo 更新elist timestamps new_labels unique_nodes   保存为npz
            elist[del_mask] = -1
            rows_to_delete = np.where(np.any(elist == -1, axis=1))  # 待删除
            elist = np.delete(elist, rows_to_delete, axis=0)
            timestamps = np.delete(timestamps, rows_to_delete, axis=0)
            timestamps = timestamps.squeeze()
            new_labels = torch.zeros(elist.shape[0])
            # edge_features = np.zeros((len(elist), 17))  # todo   初始化为0
            unique_nodes = np.unique(elist.flatten())
            # # 对于一对边，假设前面一个节点的label是这条边的lalbel，为了和FTM的源码套起来
            # for i in range(len(elist)):
            #     new_labels[i] = labels[elist[i][0]]
            #
            logger.info(f'压缩后点数:{len(save_node_l)} ，背景节点数：{len(save_bg_node_l)},目标节点数：{len(target_nodes)}')
            # print(f'压缩后点数：{len(save_node_l)},边数：{len(elist)}')
            # todo 存
            np.savez(save_path, elist=elist, timestamps=timestamps)

        # todo 取
        loaded_data = np.load(save_path)
        elist = loaded_data['elist']
        timestamps = loaded_data['timestamps']
        # new_labels = loaded_data['new_labels']
    #############################                                           todo end

    new_labels = np.zeros(elist.shape[0])
    # 对于一对边，假设前面一个节点的label是这条边的lalbel，为了和FTM的源码套起来
    for i in range(len(elist)):
        new_labels[i] = labels[elist[i][0]]
    # todo 按时间划分
    val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
    train_mask = timestamps <= val_time
    edge_features = np.zeros((len(elist), 17))  # todo   初始化为0
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    val_mask = np.where(val_mask, True, False)
    # full Data
    sources = elist[:, 0]
    destinations = elist[:, 1]
    edge_idx = np.arange(0, elist.shape[0])
    full_data = Data(sources, destinations, timestamps, edge_idx, new_labels)
    # train Data
    train_sources = sources[train_mask]
    train_destinations = destinations[train_mask]
    train_edge_idx = edge_idx[train_mask]
    train_timestamps = timestamps[train_mask]
    train_labels = new_labels[train_mask]
    train_data = Data(train_sources, train_destinations, train_timestamps, train_edge_idx, train_labels)
    # test Data
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idx[test_mask],
                     new_labels[test_mask])
    # val_mask
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idx[val_mask],
                    new_labels[val_mask])

    return node_features, edge_features, full_data, train_data, test_data, val_data


def pick_step(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)


def get_data_Dgraphfin():
    datapath = './utils/data/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False, datapath=datapath)
    print("读取数据完毕")

    return process_data(data)
