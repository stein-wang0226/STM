from scipy.io import loadmat
import numpy as np
import pandas as pd


def edgesGet(crc):
    edges = np.where((crc == 1).toarray())
    src = list(edges[0])
    des = list(edges[1])
    edges = list(zip(src, des))
    return edges

def getData(dataName):
    prefix = './utils/data/'

    data = loadmat(f'{prefix}{dataName}.mat')

    r1 = None
    r2 = None
    r3 = None
    homo = None
    if dataName == "YelpChi":
        r1 = data['net_rur']
        r2 = data['net_rtr']
        r3 = data['net_rsr']
        homo = data['homo']
    elif dataName == "Amazon":
        r1 = data['net_upu']
        r2 = data['net_usu']
        r3 = data['net_uvu']
        homo = data['homo']

    edges = edgesGet(homo)
    relation1_edges = edgesGet(r1)
    relation2_edges = edgesGet(r2)
    relation3_edges = edgesGet(r3)

    index_map = {value:index for index, value in enumerate(edges)}
    positions = [index_map.get(x, None) for x in relation1_edges]
    relation1_code = np.zeros(len(edges))
    relation1_code[positions] = 1
    positions = [index_map.get(x, None) for x in relation2_edges]
    relation2_code = np.zeros(len(edges))
    relation2_code[positions] = 1
    positions = [index_map.get(x, None) for x in relation3_edges]
    relation3_code = np.zeros(len(edges))
    relation3_code[positions] = 1
    edges_features = np.stack((relation1_code, relation2_code, relation3_code), axis=1)

    edges = np.array(edges)
    edges = np.concatenate((edges, edges_features), axis=1)

    df = pd.DataFrame(edges, columns=['i', 'j', 'relation1', 'relation2', 'relation3'])
    # 将元组中的值进行排序
    df[['i', 'j']] = np.sort(df[['i', 'j']], axis=1)
    # 使用drop_duplicates方法来删除对称元组
    filtered_df = df.drop_duplicates()
    # 将DataFrame转换为NumPy数组
    filtered_edges = filtered_df.to_numpy()
    # 随机打时间标签,总共33个时间段，每次随机取15%的节点打标签
    timestamps = np.zeros(len(filtered_edges))
    for i in range(33):
        np.random.seed(i)
        pos = np.random.randint(0, len(filtered_edges), size=int(len(filtered_edges) * 0.15))
        timestamps[pos] = i
    filtered_edges = np.concatenate((filtered_edges, timestamps.reshape(-1, 1)), axis=1)

    labels = data["label"].flatten()
    node_features = data['features'].todense().A

    filtered_edges = filtered_edges.astype(int)
    newLabelsOfSrc = np.zeros(len(filtered_edges))
    newLabelsOfDes = np.zeros(len(filtered_edges))

    for i in range(len(filtered_edges)):
        newLabelsOfSrc[i] = labels[filtered_edges[i][0]]
    for i in range(len(filtered_edges)):
        newLabelsOfDes[i] = labels[filtered_edges[i][1]]
    filtered_edges = np.concatenate((filtered_edges, newLabelsOfSrc.reshape(-1, 1)), axis=1)
    filtered_edges = np.concatenate((filtered_edges, newLabelsOfDes.reshape(-1, 1)), axis=1)

    edges_df = pd.DataFrame(filtered_edges, columns=['i', 'j', 'relation1', 'relation2', 'relation3', 't', 'srcLabel', 'desLabel'])


    edges_df.to_csv(f'{prefix}{dataName}edge_features.csv', index=False)
    print("边的形状：", edges_df.shape)

    # 进行点的特征和标签的写入
    nodes = np.concatenate((node_features, labels.reshape(-1, 1)), axis=1)
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(f'{prefix}{dataName}nodes.csv', index=False)

    print("over!")

getData("Amazon")

