import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
import torch.nn as nn



def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32
    )


class SimpleMessagePassing(MessagePassing):
    def __init__(self):
        super(SimpleMessagePassing, self).__init__(aggr="mean")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class MlpProdDecoder(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, class_num):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_num),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))





def scipy_sparse_to_torch_tensor(sparse_matrix):
    if not issparse(sparse_matrix):
        raise ValueError("Input matrix should be a SciPy sparse matrix.")
    coo_matrix = sparse_matrix.tocoo()
    indices = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)
    return indices


