import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential, SAGEConv
from torch_geometric.data import Data

class GraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        layer_sizes,
        conv_type="gcn",
        batchnorm=False,
        batchnorm_mm=0.99,
        layernorm=True,
        weight_standardization=False,
        use_feat=True,
        n_nodes=0,
        batched=False,
    ):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        assert conv_type in ["gcn", "sage", "linear"]
        
        self.conv_type = conv_type
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                if conv_type == "gcn":
                    layers.append(GCNConv(in_dim, out_dim))
                elif conv_type == "sage":
                    layers.append(SAGEConv(in_dim, out_dim))
                else:  # linear
                    layers.append(nn.Linear(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                if conv_type == "gcn":
                    layers.append((GCNConv(in_dim, out_dim), "x, edge_index -> x"))
                elif conv_type == "sage":
                    layers.append((SAGEConv(in_dim, out_dim), "x, edge_index -> x"))
                else:  # linear
                    layers.append((nn.Linear(in_dim, out_dim), "x -> x"))

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
            if conv_type in ["gcn", "sage"]:
                self.model = Sequential("x, edge_index", layers)
            else:
                self.model = Sequential("x", layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def forward(self, x, edge_index=None):
        data = Data(x, edge_index)

        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                if self.conv_type in ["gcn", "sage"]:
                    return self.model(data.x, data.edge_index)
                else:
                    return self.model(data.x)
            else:
                if self.conv_type in ["gcn", "sage"]:
                    return self.model(self.node_feats.weight.data.clone(), data.edge_index)
                else:
                    return self.model(self.node_feats.weight.data.clone())
        else:
            # batched mode
            x = data.x
            for i, conv in enumerate(self.convs):
                if self.conv_type in ["gcn", "sage"]:
                    x = conv(x, data.edge_index)
                else:
                    x = conv(x)
                x = self.relus[i](x)
                x = self.batchnorms[i](x)
            return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            conv_types = [GCNConv, SAGEConv, nn.Linear]
            if any(isinstance(m, conv_type) for conv_type in conv_types):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                if hasattr(m, 'lin'):
                    weight = m.lin.weight.data
                else:
                    weight = m.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                if hasattr(m, 'lin'):
                    m.lin.weight.data = weight
                else:
                    m.weight.data = weight

    @property
    def num_layers(self):
        return self.n_layers

