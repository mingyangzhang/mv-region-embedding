import numpy as np
import scipy.io as sio
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    """ Return adjs. """

    data_path = "../data/"
    adj = np.load(data_path+"mob-adj.npy")
    _, n, _  = adj.shape

    adj = adj/np.mean(adj, axis=(1, 2))

    k = 20
    s_adj = np.load(data_path+"s_adj.npy")
    s_adj_sp = np.copy(adj[0])

    for i in range(n):
        s_adj_sp[np.argsort(s_adj_sp[:, i])[:-k], i] = 0
        s_adj_sp[i, np.argsort(s_adj_sp[i, :])[:-k]] = 0
        # s_adj[i, np.argsort(s_adj[i, :])[:-k]] = 0

    t_adj = np.load(data_path+"t_adj.npy")
    t_adj_sp = np.copy(adj[0])

    for i in range(n):
        t_adj_sp[np.argsort(t_adj_sp[:, i])[:-k], i] = 0
        t_adj_sp[i, np.argsort(t_adj_sp[i, :])[:-k]] = 0

    k = 15
    poi_adj = np.load(data_path+"poi_simi.npy")
    poi_adj_sp = np.copy(poi_adj)
    for i in range(n):
        poi_adj_sp[np.argsort(poi_adj_sp[:, i])[:-k], i] = 0
        poi_adj_sp[i, np.argsort(poi_adj_sp[i, :])[:-k]] = 0

    chk_adj = np.load(data_path+"chk_simi.npy")
    chk_adj[np.isnan(chk_adj)] = 0
    chk_adj_sp = np.copy(chk_adj)
    for i in range(n):
        chk_adj_sp[np.argsort(chk_adj_sp[:, i])[:-k], i] = 0
        chk_adj_sp[i, np.argsort(chk_adj_sp[i, :])[:-k]] = 0


    feature = np.random.uniform(-1, 1, size=(180, 250))
    feature = feature[np.newaxis]

    out = {

        "mob_adj": adj,
        "s_adj_sp": s_adj_sp,
        "t_adj_sp": t_adj_sp,
        "poi_adj": poi_adj,
        "poi_adj_sp": poi_adj_sp,
        "chk_adj": chk_adj,
        "chk_adj_sp": chk_adj_sp,
        "feature": feature,
    }

    return out

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(feature):
    """Row-normalize feature matrix and convert to tuple representation"""
    feature = feature[0]
    colvar = np.var(feature, axis=1, keepdims=True)
    colmean = np.mean(feature, axis=1, keepdims=True)
    c_inv = np.power(colvar, -0.5)
    c_inv[np.isinf(c_inv)] = 0.
    feature = np.multiply((feature - colmean), c_inv)
    feature = feature[np.newaxis]
    return feature


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(data, placeholders):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update({placeholders['feature']: data['feature']})
    feed_dict.update({placeholders['mob_adj']: data['mob_adj']})
    feed_dict.update({placeholders['poi_adj']: data['poi_adj']})
    feed_dict.update({placeholders['chk_adj']: data['chk_adj']})

    feed_dict.update({placeholders['s_bias']: data['s_bias']})
    feed_dict.update({placeholders['t_bias']: data['t_bias']})
    feed_dict.update({placeholders['poi_bias']: data['poi_bias']})
    feed_dict.update({placeholders['chk_bias']: data['chk_bias']})

    return feed_dict

def adj_to_bias(adj, sizes, nhood=1):
    adj = adj[np.newaxis]
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def adj_to_mask(adj, k):
    n = adj.shape[0]
    mask = np.zeros((n, n, 2))
    for i in range(n):
        mask[i, np.argsort(adj[i, :])[:k], 1] = 1
        mask[i, np.argsort(adj[i, :])[-k:], 0] = 1
    return mask