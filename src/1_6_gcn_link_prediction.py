import argparse

from tqdm import tqdm
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Flickr,
    CitationFull,
    CoraFull,
    Reddit,
    Coauthor,
)

from utils import (
    load_data,
    load_data_social,
    accuracy,
    mask_test_edges,
    preprocess_graph,
    loss_function_gcn,
    get_roc_score_GCN,
    torch_sparse_tensor_to_sparse_mx,
    do_transductive_edge_split,
)
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

from scipy.sparse import issparse
import argparse
import scipy.sparse as sp
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from os import path
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Union
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential, SAGEConv
from torch_geometric.data import Data

from torch_geometric.utils import negative_sampling


# import warnings
# warnings.filterwarnings("ignore")


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA training.")
parser.add_argument(
    "--dataset",
    type=str,
    default="Flickr",
    help="One dataset from \
Cora, CiteSeer, PubMed, CoraFull, Amazon-Computers, Amazon-Photo, \
CitationFull-DBLP, Flickr, Reddit, Coauthor-CS, Coauthor-Physics.",
)

parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."  # 200
)
parser.add_argument("--p", type=float, default=1.0, help="Initial learning rate.")
parser.add_argument("--q", type=float, default=1.0, help="Initial learning rate.")

parser.add_argument(
    "--num_of_nodes", type=int, default=-1, help="Write reuslts in files."
)

parser.add_argument(
    "--rank",
    type=int,
    default=1024,
    help="The bound of the rank for the learnable embedding matrix.",
)

parser.add_argument(
    "--embedding_dim", type=int, default=1024, help="Dimension of embeddings."
)


parser.add_argument(
    "--epoch_threshold", type=int, default=100, help="Dimension of embeddings."  # 100
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="The bound of the rank for the learnable embedding matrix.",
)


parser.add_argument("--write", type=int, default=1, help="Write reuslts in files.")

parser.add_argument("--cluster", type=int, default=0, help="Dimension of embeddings.")

parser.add_argument("--comp_idx", type=int, default=-1, help="Dimension of embeddings.")


parser.add_argument(
    "--feature_dim",
    type=float,
    default=1.0,
    help="The bound of the rank for the learnable embedding matrix.",
)


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
dataset_name = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cpu"
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = "cuda"


def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32
    )


class MlpProdDecoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        left, right = x[:, : self.embedding_size], x[:, self.embedding_size :]
        return self.net(left * right)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class GCN(nn.Module):
    def __init__(
        self,
        layer_sizes,
        batchnorm=False,
        batchnorm_mm=0.99,
        layernorm=True,
        weight_standardization=False,
        use_feat=True,
        n_nodes=0,
        batched=False,
        input_feature_dim=-1,
    ):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                layers.append(GCNConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(
                    (GCNConv(in_dim, out_dim), "x, edge_index -> x"),
                )

                if batchnorm:
                    layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
                else:
                    layers.append(LayerNorm(out_dim))

                layers.append(nn.PReLU())

        if batched:
            self.convs = nn.ModuleList(layers)
            self.relus = nn.ModuleList(relus)
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            self.model = Sequential("x, edge_index", layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def forward(self, x, edge_index):
        data = Data(x, edge_index)

        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                return self.model(data.x, data.edge_index)
            return self.model(self.node_feats.weight.data.clone(), data.edge_index)
        # otherwise, batched
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = self.relus[i](x)
            x = self.batchnorms[i](x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

    def get_node_feats(self):
        if hasattr(self, "node_feats"):
            return self.node_feats
        return None

    @property
    def num_layers(self):
        return self.n_layers


class Node2Vec(torch.nn.Module):
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        rank_bound: int = 16,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
        input_feature_dim=-1,
    ):
        super().__init__()

        if p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        else:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.rank_bound = rank_bound

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.gnn = GCN([input_feature_dim, self.rank_bound], n_nodes=num_nodes).to(
            device
        )

    def forward(self, batch: Optional[Tensor] = None, trans=None) -> Tensor:
        emb = self.gnn(data.x.to(device), data.edge_index.to(device))
        if trans != None:
            emb = torch.matmul(emb, trans)

        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(
            self.rowptr, self.col, batch, self.walk_length, self.p, self.q
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(
            self.num_nodes,
            (batch.size(0), self.walk_length),
            dtype=batch.dtype,
            device=batch.device,
        )
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor, trans=None) -> Tensor:
        embeddings = self.gnn(data.x.to(device), data.edge_index.to(device))
        if trans != None:
            embeddings = torch.matmul(embeddings, trans)

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = embeddings[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = embeddings[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = "lbfgs",  # lbfgs
        multi_class: str = "auto",
        *args,
        **kwargs,
    ) -> float:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args, **kwargs
        ).fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        macro_f1 = f1_score(
            test_y.detach().cpu().numpy(),
            clf.predict(test_z.detach().cpu().numpy()),
            average="macro",
        )
        micro_f1 = f1_score(
            test_y.detach().cpu().numpy(),
            clf.predict(test_z.detach().cpu().numpy()),
            average="micro",
        )
        auc = roc_auc_score(
            test_y.detach().cpu().numpy(),
            clf.predict_proba(test_z.detach().cpu().numpy()),
            multi_class="ovr",
        )

        return (
            clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy()),
            macro_f1,
            micro_f1,
            auc,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.embedding.size(0)}, "
            f"{self.embedding.size(1)})"
        )


def get_roc_score(emb, edges_pos, edges_neg):
    def sigmoid(x):
        x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    preds = []

    for e in edges_pos:
        preds.append(sigmoid((emb[e[0], :] * emb[e[1], :]).sum()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid((emb[e[0], :] * emb[e[1], :]).sum()))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, preds_all


def scipy_sparse_to_torch_tensor(sparse_matrix):
    if not issparse(sparse_matrix):
        raise ValueError("Input matrix should be a SciPy sparse matrix.")

    coo_matrix = sparse_matrix.tocoo()
    indices = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)
    return indices


def eval_hits(y_pred_pos, y_pred_neg, K):
    if len(y_pred_neg) < K:
        log.warn(f"[WARNING]: hits@{K} defaulted to 1")
        return {"hits@{}".format(K): 1.0}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
        y_pred_pos
    )
    return hitsK


def low_degree_metrics(preds_all_test, global_graph, test_edges, test_edges_neg):
    test_edges_neg = np.array(test_edges_neg)

    preds_all_test = preds_all_test.astype(np.float64)

    preds_all_test_torch = torch.from_numpy(preds_all_test)

    # normal hitrates

    test_edge_pos_num = test_edges.shape[0]

    hits10 = eval_hits(
        preds_all_test_torch[:test_edge_pos_num],
        preds_all_test_torch[test_edge_pos_num:],
        10,
    )
    hits20 = eval_hits(
        preds_all_test_torch[:test_edge_pos_num],
        preds_all_test_torch[test_edge_pos_num:],
        20,
    )
    hits30 = eval_hits(
        preds_all_test_torch[:test_edge_pos_num],
        preds_all_test_torch[test_edge_pos_num:],
        30,
    )
    hits40 = eval_hits(
        preds_all_test_torch[:test_edge_pos_num],
        preds_all_test_torch[test_edge_pos_num:],
        40,
    )
    hits50 = eval_hits(
        preds_all_test_torch[:test_edge_pos_num],
        preds_all_test_torch[test_edge_pos_num:],
        50,
    )

    print("hits50", hits50)

    # low degree nodes
    preds_all = preds_all_test
    labels_all = np.hstack(
        [np.ones(test_edges.shape[0]), np.zeros(test_edges_neg.shape[0])]
    )

    the_concatenation = np.concatenate((test_edges.flatten(), test_edges_neg.flatten()))

    degrees = np.array([global_graph.degree(node) for node in the_concatenation])

    degree_ranks = np.argsort(np.argsort(degrees))

    position_marker = degree_ranks < (degree_ranks.shape[0] / 2.0)

    low_degree_nodes = the_concatenation.flatten()[position_marker]

    low_degree_marker = []

    for edge in np.concatenate((test_edges, np.array(test_edges_neg)), axis=0):
        if edge[0] in low_degree_nodes and edge[1] in low_degree_nodes:
            low_degree_marker.append(True)
        else:
            low_degree_marker.append(False)

    roc_score_low_degree = roc_auc_score(
        labels_all[low_degree_marker], preds_all[low_degree_marker]
    )
    ap_score_low_degree = average_precision_score(
        labels_all[low_degree_marker], preds_all[low_degree_marker]
    )

    truncation_marker = low_degree_marker[: test_edges.shape[0]]

    hits10_low_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        10,
    )
    hits20_low_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        20,
    )
    hits30_low_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        30,
    )
    hits40_low_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        40,
    )
    hits50_low_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        50,
    )

    # high degree nodes

    position_marker = degree_ranks > (degree_ranks.shape[0] / 2.0)

    low_degree_nodes = the_concatenation.flatten()[position_marker]

    low_degree_marker = []

    for edge in np.concatenate((test_edges, np.array(test_edges_neg)), axis=0):
        if edge[0] in low_degree_nodes and edge[1] in low_degree_nodes:
            low_degree_marker.append(True)
        else:
            low_degree_marker.append(False)

    roc_score_high_degree = roc_auc_score(
        labels_all[low_degree_marker], preds_all[low_degree_marker]
    )
    ap_score_high_degree = average_precision_score(
        labels_all[low_degree_marker], preds_all[low_degree_marker]
    )

    truncation_marker = low_degree_marker[: test_edges.shape[0]]

    hits10_high_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        10,
    )
    hits20_high_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        20,
    )
    hits30_high_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        30,
    )
    hits40_high_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        40,
    )
    hits50_high_degree = eval_hits(
        preds_all_test_torch[:test_edge_pos_num][truncation_marker],
        preds_all_test_torch[test_edge_pos_num:],
        50,
    )

    return {
        "hits10": hits10,
        "hits20": hits20,
        "hits30": hits30,
        "hits40": hits40,
        "hits50": hits50,
        "roc_score_low_degree": roc_score_low_degree,
        "ap_score_low_degree": ap_score_low_degree,
        "hits10_low_degree": hits10_low_degree,
        "hits20_low_degree": hits20_low_degree,
        "hits30_low_degree": hits30_low_degree,
        "hits40_low_degree": hits40_low_degree,
        "hits50_low_degree": hits50_low_degree,
        "roc_score_high_degree": roc_score_high_degree,
        "ap_score_high_degree": ap_score_high_degree,
        "hits10_high_degree": hits10_high_degree,
        "hits20_high_degree": hits20_high_degree,
        "hits30_high_degree": hits30_high_degree,
        "hits40_high_degree": hits40_high_degree,
        "hits50_high_degree": hits50_high_degree,
    }


def build_graph_with_max_nodes(edges, max_nodes):
    edgelist = [tuple(edge) for edge in edges.cpu().numpy().T]

    G = nx.from_edgelist(edgelist)

    num_nodes = G.number_of_nodes()

    existing_nodes = set(G.nodes())
    missing_nodes = set(range(max_nodes)) - existing_nodes

    for node in missing_nodes:
        G.add_node(node)

    return G


if __name__ == "__main__":
    # load data with args

    assert args.embedding_dim >= args.rank

    def dealing_with_edges(data):
        weights = np.ones(data.edge_index.shape[1])  # .astype(int)

        adj = sp.csr_matrix(
            (weights, (data.edge_index[0, :], data.edge_index[1, :])),
            shape=(data.x.shape[0], data.x.shape[0]),
        )

        (
            adj_train,
            train_edges,
            val_edges,
            val_edges_false,
            test_edges,
            test_edges_false,
        ) = do_transductive_edge_split(adj, data)

        data.edge_index = torch.LongTensor(scipy_sparse_to_torch_tensor(adj_train))

        return data, val_edges, val_edges_false, test_edges, test_edges_false

    dataset = None
    data = None

    print("Loading dataset ... ")
    if args.dataset == "Cora":
        dataset = Planetoid(root="../data/Cora", name="Cora")
        data = dataset[0]
    elif args.dataset == "CiteSeer":
        dataset = Planetoid(root="../data/CiteSeer", name="CiteSeer")
        data = dataset[0]
    elif args.dataset == "PubMed":
        dataset = Planetoid(root="../data/PubMed", name="PubMed")
        data = dataset[0]
    elif args.dataset == "CoraFull":
        dataset = CoraFull(root="../data/CoraFull")
        data = dataset[0]
    elif args.dataset == "Amazon-Computers":
        dataset = Amazon(root="../data/Amazon-Computers", name="Computers")
        data = dataset[0]
    elif args.dataset == "Amazon-Photo":
        dataset = Amazon(root="../data/Amazon-Photo", name="Photo")
        data = dataset[0]
    elif args.dataset == "CitationFull-Cora":
        dataset = CitationFull(root="../data/CitationFull-Cora", name="Cora")
        data = dataset[0]
    elif args.dataset == "CitationFull-DBLP":
        dataset = CitationFull(root="../data/CitationFull-DBLP", name="DBLP")
        data = dataset[0]
    elif args.dataset == "Flickr":
        dataset = Flickr(root="../data/Flickr")
        data = dataset[0]
    elif args.dataset == "Reddit":
        dataset = Reddit(root="../data/Reddit")
        data = dataset[0]
    elif args.dataset == "Coauthor-CS":
        dataset = Coauthor(root="../data/Coauthor-CS", name="CS")
        data = dataset[0]
    elif args.dataset == "Coauthor-Physics":
        dataset = Coauthor(root="../data/Coauthor-Physics", name="Physics")
        data = dataset[0]
        V = torch.pca_lowrank(data.x, q=1000)[2].to(device)
        data.x = torch.matmul(data.x.to(device), V)

    print("Dataset loaded. ")

    nodes = data.edge_index.t().numpy()
    nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))
    args.num_of_nodes = nodes.max() + 1

    feature_using = int(data.x.shape[1] * args.feature_dim)
    data.x = data.x[:, : max(1, feature_using)]

    print("Using feature:")
    print(args.feature_dim)
    print(max(1, feature_using))

    data, val_edges, val_edges_false, test_edges, test_edges_false = dealing_with_edges(
        data
    )

    data.x = F.normalize(data.x.to(device), p=1, dim=1)

    model = Node2Vec(
        data.edge_index,
        embedding_dim=args.embedding_dim,
        walk_length=20,
        context_size=10,
        rank_bound=args.rank,
        walks_per_node=10,
        num_negative_samples=1,
        p=args.p,
        q=args.q,
        sparse=True,
        num_nodes=args.num_of_nodes,
        input_feature_dim=data.x.shape[1],
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)  # 0.01

    transition_matrix_v = 1e-1 * torch.eye(args.rank, args.embedding_dim).to(device)

    transition_matrix_v.requires_grad = False

    def train(epoch):
        if epoch < args.epoch_threshold:
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(
                    pos_rw.to(device), neg_rw.to(device), trans=transition_matrix_v
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)
        else:
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer2.zero_grad()
                loss = model.loss(
                    pos_rw.to(device), neg_rw.to(device), trans=transition_matrix_v
                )
                loss.backward()
                optimizer2.step()
                total_loss += loss.item()
            return total_loss / len(loader)

    def train_an_mlp(
        model_out, train_edge_label_index, train_edge_label, test_edge_label_index
    ):
        predictor = MlpProdDecoder(model_out.shape[1], hidden_size=256).to(device)
        criterion = BCEWithLogitsLoss()

        # optimizer
        optimizer = Adam(list(predictor.parameters()), lr=5e-3)

        for i in range(100):
            predictor.train()
            optimizer.zero_grad()

            edge_embeddings = model_out[train_edge_label_index]
            combined = torch.hstack(
                (edge_embeddings[0, :, :], edge_embeddings[1, :, :])
            )
            out = predictor(combined)

            loss = criterion(
                out.view(-1).to(device), train_edge_label.float().to(device)
            )
            loss.backward()
            optimizer.step()

        predictor.eval()

        edge_embeddings = model_out[test_edge_label_index]
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        out = predictor(combined)

        return out

    def test(val_edges, val_edges_false, test_edges, test_edges_false, flag=0):
        model.eval()

        z = model(trans=transition_matrix_v)

        model_out = z.detach()

        train_edge = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=train_edge,
            num_nodes=data.x.shape[0],
            num_neg_samples=train_edge.size(1),
            method="sparse",
        )

        adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            value=None,
            sparse_sizes=(data.x.shape[0], data.x.shape[0]),
        )

        adj_t = adj.t()
        row, col, _ = adj_t.coo()

        train_edge_label_index = torch.cat(
            [train_edge, neg_edge_index],
            dim=-1,
        )
        train_edge_label = torch.cat(
            [
                train_edge.new_ones(train_edge.size(1)),
                train_edge.new_zeros(neg_edge_index.size(1)),
            ],
            dim=0,
        )

        val_edges_array = np.array(val_edges).T
        val_edges_false_array = np.array(val_edges_false).T

        val_edge_label_index = torch.cat(
            [
                torch.from_numpy(val_edges_array).to(device),
                torch.from_numpy(val_edges_false_array).to(device),
            ],
            dim=-1,
        )
        val_edge_label = torch.cat(
            [
                data.edge_index.new_ones(val_edges_array.shape[1]),
                data.edge_index.new_zeros(val_edges_false_array.shape[1]),
            ],
            dim=0,
        )

        test_edges_array = np.array(test_edges).T
        test_edges_false_array = np.array(test_edges_false).T

        test_edge_label_index = torch.cat(
            [
                torch.from_numpy(test_edges_array).to(device),
                torch.from_numpy(test_edges_false_array).to(device),
            ],
            dim=-1,
        )
        test_edge_label = torch.cat(
            [
                data.edge_index.new_ones(test_edges_array.shape[1]),
                data.edge_index.new_zeros(test_edges_false_array.shape[1]),
            ],
            dim=0,
        )

        out = train_an_mlp(
            model_out.detach(),
            train_edge_label_index,
            train_edge_label,
            val_edge_label_index,
        )

        roc_score_val = roc_auc_score(
            val_edge_label.float().detach().numpy(), out.view(-1).detach().cpu().numpy()
        )
        ap_score_val = average_precision_score(
            val_edge_label.float().detach().numpy(), out.view(-1).detach().cpu().numpy()
        )

        out = train_an_mlp(
            model_out.detach(),
            train_edge_label_index,
            train_edge_label,
            test_edge_label_index,
        )

        roc_score_test = roc_auc_score(
            test_edge_label.float().detach().numpy(),
            out.view(-1).detach().cpu().numpy(),
        )
        ap_score_test = average_precision_score(
            test_edge_label.float().detach().numpy(),
            out.view(-1).detach().cpu().numpy(),
        )

        print("roc_score_test", roc_score_test)
        print("ap_score_test", ap_score_test)

        rank = 0.000

        if flag == 1:
            if args.cluster == 1 and dataset != "Reddit" and dataset != "Flickr":
                rank = np.linalg.matrix_rank(model_out.detach().cpu().numpy())
            elif args.cluster == 0:
                rank = torch.linalg.matrix_rank(model_out.detach())

        return (
            roc_score_val,
            ap_score_val,
            roc_score_test,
            ap_score_test,
            rank,
            out.flatten().detach().cpu().numpy(),
        )

    for epoch in tqdm(range(0, args.epochs)):
        loss = train(epoch)

        if epoch == args.epoch_threshold - 1:
            transition_matrix_v = 1e-1 * (
                torch.eye(args.rank, args.embedding_dim)
                + torch.randn(args.rank, args.embedding_dim) / 1e4
            ).to(device)
            transition_matrix_v.requires_grad = True

            parameters_v = [{"params": [transition_matrix_v]}]
            # optimizer2
            optimizer2 = torch.optim.Adam(parameters_v, lr=0.001, weight_decay=1e-4)

            for param in model.parameters():
                param.requires_grad = False

    (
        roc_score_val,
        ap_score_val,
        roc_score_test,
        ap_score_test,
        rank,
        preds_all_test,
    ) = test(val_edges, val_edges_false, test_edges, test_edges_false, flag=1)
    print(
        f"Epoch: {epoch:02d}, Loss: {loss:.4f}, roc_score_val: {roc_score_val:.4f}, ap_score_val: {ap_score_val:.4f}, roc_score_test: {roc_score_test:.4f}, ap_score_test: {ap_score_test:.4f}, rank: {rank:.4f}"
    )

    global_graph = build_graph_with_max_nodes(data.edge_index, args.num_of_nodes)

    hits_and_degree_bias = low_degree_metrics(
        preds_all_test, global_graph, test_edges, test_edges_false
    )

    print(hits_and_degree_bias)

    comp = list(hits_and_degree_bias.values())

    formatted_comp = [f"{num:.4f}" for num in comp]

    comp_string = ",".join(formatted_comp)

    prediction_dict = {
        "predicted": preds_all_test,
        "num_of_positives": test_edges.shape[0],
        "test_edges": test_edges,
        "test_edges_false": test_edges_false,
    }

    if args.write == 1:
        if args.cluster == 1:
            assert 1 == 0
            # if not os.path.exists("../../1_results/1_link_pred_completed.txt"):
            #     print(f"1_link_pred_completed.txt does not exist, creating it...")

            #     with open("../../1_results/1_link_pred_completed.txt", "a") as f:
            #         formatted_comp_keys = list(hits_and_degree_bias.keys())
            #         comp_string_keys = ",".join(formatted_comp_keys)
            #         f.write(
            #             "model,dataset,roc_score_val,ap_score_val,roc_score_test,ap_score_test,rank,rank_bound,embedding_dim,p,q,feature_dim,seed,"
            #             + comp_string_keys
            #             + ",compidx\n"
            #         )

            # with open(f"../../1_results/1_link_pred_completed.txt", "a") as file:
            #     line = (
            #         f"gcn,{args.dataset},{roc_score_val:.4f},{ap_score_val:.4f},{roc_score_test:.4f},{ap_score_test:.4f},{rank:.1f},{args.rank:.1f},{args.embedding_dim:.1f},{args.p:.1f},{args.q:.1f},{args.feature_dim:.8f},{args.seed:.1f},"
            #         + comp_string
            #         + ","
            #         + str(args.comp_idx)
            #         + "\n"
            #     )
            #     file.write(line)

            # np.save(
            #     f"../../1_link_prediction/gcn_{args.dataset}_{args.rank:.0f}_{args.embedding_dim:.0f}_{args.p:.0f}_{args.q:.0f}_{args.feature_dim:.8f}_{args.seed:.0f}.npy",
            #     prediction_dict,
            # )

        elif args.cluster == 0:
            if not os.path.exists("../example_results/1_link_pred_completed.txt"):
                print(f"1_link_pred_completed.txt does not exist, creating it...")

                with open("../example_results/1_link_pred_completed.txt", "a") as f:
                    formatted_comp_keys = list(hits_and_degree_bias.keys())
                    comp_string_keys = ",".join(formatted_comp_keys)
                    f.write(
                        "model,dataset,roc_score_val,ap_score_val,roc_score_test,ap_score_test,rank,rank_bound,embedding_dim,p,q,feature_dim,seed,"
                        + comp_string_keys
                        + ",compidx\n"
                    )

            with open(f"../example_results/1_link_pred_completed.txt", "a") as file:
                line = (
                    f"gcn,{args.dataset},{roc_score_val:.4f},{ap_score_val:.4f},{roc_score_test:.4f},{ap_score_test:.4f},{rank:.1f},{args.rank:.1f},{args.embedding_dim:.1f},{args.p:.1f},{args.q:.1f},{args.feature_dim:.8f},{args.seed:.1f},"
                    + comp_string
                    + ","
                    + str(args.comp_idx)
                    + "\n"
                )
                file.write(line)

            np.save(
                f"../example_results/gcn_lp_{args.dataset}_{args.rank:.0f}_{args.embedding_dim:.0f}_{args.p:.0f}_{args.q:.0f}_{args.feature_dim:.8f}_{args.seed:.0f}.npy",
                prediction_dict,
            )
