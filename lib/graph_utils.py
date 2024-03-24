import os
import math
import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from .utils import log_string
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def construct_tem_adj(data, num_node):
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node)):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]

    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0]):
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0])+1] # NlogN edges
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T)
    return tem_matrix

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_adj_matrix = np.dot(D, W)

    return norm_adj_matrix
def get_norm_adj(spatial_graph):
    adj_mx = np.load(spatial_graph)
    norm_adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor)
    return norm_adj_matrix


def loadGraph(spatial_graph, temporal_graph, dims, data, log):
    # calculate spatial and temporal graph wavelets
    adj = np.load(spatial_graph)
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph):
        tem_adj = np.load(temporal_graph)
    else:
        # 生成temporal_graph的图结构temadj.npz
        # construct_tem_adj采用DTW方法构建时间邻接矩阵(N, N)，来表示不同时间序列之间的相似性。
        tem_adj = construct_tem_adj(data, adj.shape[0])
        np.save(temporal_graph, tem_adj)
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(tem_adj, dims)
    # spawave, temwave分别表示空间图和时间图的特征值与特征向量
    # 时间图描绘了不同时间点的交通流量之间的相似性
    log_string(log, f'Shape of graphwave eigenvalue and eigenvector: {spawave[0].shape}, {spawave[1].shape}')

    # derive neighbors
    # 这行代码计算每个节点应该采样的邻居数量。这里使用了对数规模来确定邻居的数量
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    # 采用csr格式存储稀疏矩阵
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    # 通过迪杰斯特拉算法得到每个节点最近的sampled_nodes_number个邻居
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    log_string(log, f'Shape of localadj: {localadj.shape}')
    return localadj, spawave, temwave
