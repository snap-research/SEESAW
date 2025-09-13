from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

from src.gnn import GraphNeuralNetwork
from src.tools import index2ptr, SimpleMessagePassing



class UnifiedGraphModel(torch.nn.Module):
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
        input_feature_dim: int = -1,
        model_type: str = "deepwalk",
        device: str = "cpu",
    ):
        super().__init__()

        # Model type validation
        assert model_type in ["deepwalk", "gcn", "sage", "deepwalk_prop", "gnn_no_prop"]
        
        self.model_type = model_type
        self.device = device
        
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
        
        # Initialize components based on model type
        if self.model_type == "deepwalk":
            # Pure deepwalk with low-rank decomposition
            self.U = Embedding(self.num_nodes, self.rank_bound).to(device)
            self.V = Embedding(self.rank_bound, self.embedding_dim).to(device)
            
        elif self.model_type == "deepwalk_prop":
            # Deepwalk with message passing
            self.U = Embedding(self.num_nodes, self.rank_bound).to(device)
            self.V = Embedding(self.rank_bound, self.embedding_dim).to(device)
            self.model_message_passing = SimpleMessagePassing().to(device)
            self.edge_index = edge_index.to(device)
            
        elif self.model_type in ["gcn", "sage", "gnn_no_prop"]:
            # GNN-based models
            conv_type = "linear" if self.model_type == "gnn_no_prop" else self.model_type
            self.gnn = GraphNeuralNetwork(
                [input_feature_dim, self.rank_bound], 
                conv_type=conv_type,
                n_nodes=num_nodes
            ).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        if self.model_type in ["deepwalk", "deepwalk_prop"]:
            self.U.reset_parameters()
            transition_matrix_v = torch.eye(self.rank_bound, self.embedding_dim).to(self.device)
            self.V.weight = torch.nn.Parameter(transition_matrix_v)
            self.V.weight.requires_grad = False

    def forward(self, batch: Optional[Tensor] = None, test=0, trans=None) -> Tensor:
        if self.model_type == "deepwalk":
            embeddings = torch.matmul(self.U.weight, self.V.weight)
            return embeddings if batch is None else embeddings.index_select(0, batch)
            
        elif self.model_type == "deepwalk_prop":
            embeddings = torch.matmul(self.U.weight, self.V.weight)
            embeddings = self.model_message_passing(embeddings, self.edge_index)
            return embeddings if batch is None else embeddings.index_select(0, batch)
            
        elif self.model_type in ["gcn", "sage", "gnn_no_prop"]:
            # Access global data - this needs to be set externally
            emb = self.gnn(self.data.x.to(self.device), self.data.edge_index.to(self.device))
            if trans is not None:
                emb = torch.matmul(emb, trans)
            return emb if batch is None else emb.index_select(0, batch)

    def set_data(self, data):
        """Set the data object for GNN models"""
        self.data = data

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
    def loss(self, pos_rw: Tensor, neg_rw: Tensor, epoch=None, trans=None) -> Tensor:
        if self.model_type == "deepwalk":
            embeddings = torch.matmul(self.U.weight, self.V.weight)
        elif self.model_type == "deepwalk_prop":
            embeddings = torch.matmul(self.U.weight, self.V.weight)
        elif self.model_type in ["gcn", "sage", "gnn_no_prop"]:
            embeddings = self.gnn(self.data.x.to(self.device), self.data.edge_index.to(self.device))
            if trans is not None:
                embeddings = torch.matmul(embeddings, trans)

        # Positive loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = embeddings[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = embeddings[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = embeddings[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss
