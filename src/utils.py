import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


from numpy import inf
import scipy.io as scio
import pandas as pd
import numpy as np
import pickle as pkl
import networkx as nx
from numpy import inf

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import (
    negative_sampling,
    add_self_loops,
    train_test_split_edges,
    subgraph,
)
import math
from absl import flags


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()
    sp_matrix = sp.coo_matrix(
        (data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1])
    )
    return sp_matrix


def loss_function_gcn(preds, labels, pos_weight):
    cost = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    return cost


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    )
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def get_roc_score_GCN(adj, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = adj
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    # # 85 / 5 / 10
    num_test = int(np.floor(edges.shape[0] / 10.0))
    num_val = int(np.floor(edges.shape[0] / 20.0))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val : (num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    print("started generating " + str(num_test) + " negative test edges...")

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    print("started generating " + str(num_val) + " negative validation edges...")
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape
    )
    adj_train = adj_train + adj_train.T

    return (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    )


def do_transductive_edge_split(
    adj, data, fast_split=False, val_ratio=0.05, test_ratio=0.1, split_seed=234
):
    num_nodes = adj.shape[0]
    row, col = data.edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0).t()
    r, c = row[n_v : n_v + n_t], col[n_v : n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0).t()
    r, c = row[n_v + n_t :], col[n_v + n_t :]
    data.train_pos_edge_index = torch.stack([r, c], dim=0).t()
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(
        data.edge_index, num_nodes=num_nodes, num_neg_samples=row.size(0)
    )
    data.val_neg_edge_index = neg_edge_index[:, :n_v].t()
    data.test_neg_edge_index = neg_edge_index[:, n_v : n_v + n_t].t()
    data.train_neg_edge_index = neg_edge_index[:, n_v + n_t :].t()

    print(data.train_pos_edge_index.shape)
    print(data.val_pos_edge_index.shape)
    print(data.val_neg_edge_index.shape)
    print(data.test_pos_edge_index.shape)
    print(data.test_neg_edge_index.shape)

    # Re-build adj matrix
    entries = np.ones(data.train_pos_edge_index.shape[0])
    adj_train = sp.csr_matrix(
        (entries, (data.train_pos_edge_index[:, 0], data.train_pos_edge_index[:, 1])),
        shape=adj.shape,
    )
    adj_train = adj_train + adj_train.T

    return (
        adj_train,
        data.train_pos_edge_index,
        data.val_pos_edge_index,
        data.val_neg_edge_index,
        data.test_pos_edge_index,
        data.test_neg_edge_index,
    )


def load_data_social(dataset="BlogCatalog", mode="s"):
    print("Loading {} dataset...".format(dataset))
    dataFile = "../data/" + dataset + "/" + dataset + ".mat"
    data = scio.loadmat(dataFile)
    labels = encode_onehot(list(data["Label"][:, 0]))
    adj = sp.csr_matrix(data["Network"].toarray()[:, :])
    features = data["Attributes"].toarray()[:, :]
    print("Dataset has {} nodes, {} features.".format(adj.shape[0], features.shape[1]))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = D - adj

    if mode == "s":
        with np.errstate(divide="ignore"):
            D_norm = D ** (-0.5)
        D_norm[D_norm == inf] = 0
        adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))
    elif mode == "r":
        with np.errstate(divide="ignore"):
            D_norm = np.linalg.inv(D)
        adj = sp.coo_matrix(D_norm.dot(l))

    list_split = []
    length_of_data = adj.shape[0]
    train_percent = 0.1
    val_percent = 0.2

    for i in range(length_of_data):
        list_split.append(i)

    node_perm = np.random.permutation(labels.shape[0])
    idx_train = node_perm[: int(train_percent * length_of_data)]  # list_split
    idx_val = node_perm[
        int(train_percent * length_of_data) : int(
            train_percent * length_of_data + val_percent * length_of_data
        )
    ]
    idx_test = node_perm[
        int(train_percent * length_of_data + val_percent * length_of_data) :
    ]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
