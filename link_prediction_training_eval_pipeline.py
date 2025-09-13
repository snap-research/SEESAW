from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from torch_geometric.datasets import (
    Planetoid, Amazon, Flickr, CitationFull, CoraFull, Reddit, Coauthor,
)
import networkx as nx
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split

from src.unified_graph_model import UnifiedGraphModel
from src.tools import MlpProdDecoder


class LinkPredictionMLP(torch.nn.Module):
    """Modified MLP for link prediction (product of embeddings)"""
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        left, right = x[:, : self.embedding_size], x[:, self.embedding_size :]
        return self.net(left * right)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class LinkPredictionTrainingPipeline:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = None
        self.optimizer = None
        self.optimizer2 = None
        self.transition_matrix_v = None
        
    def load_dataset(self):
        """Load and preprocess dataset"""
        print("Loading dataset...")
        
        if self.args.dataset == "Cora":
            dataset = Planetoid(root="./data/Cora", name="Cora")
            data = dataset[0]
        elif self.args.dataset == "CiteSeer":
            dataset = Planetoid(root="./data/CiteSeer", name="CiteSeer")
            data = dataset[0]
        elif self.args.dataset == "PubMed":
            dataset = Planetoid(root="./data/PubMed", name="PubMed")
            data = dataset[0]
        elif self.args.dataset == "CoraFull":
            dataset = CoraFull(root="./data/CoraFull")
            data = dataset[0]
        elif self.args.dataset == "Amazon-Computers":
            dataset = Amazon(root="./data/Amazon-Computers", name="Computers")
            data = dataset[0]
        elif self.args.dataset == "Amazon-Photo":
            dataset = Amazon(root="./data/Amazon-Photo", name="Photo")
            data = dataset[0]
        elif self.args.dataset == "CitationFull-Cora":
            dataset = CitationFull(root="./data/CitationFull-Cora", name="Cora")
            data = dataset[0]
        elif self.args.dataset == "CitationFull-DBLP":
            dataset = CitationFull(root="./data/CitationFull-DBLP", name="DBLP")
            data = dataset[0]
        elif self.args.dataset == "Flickr":
            dataset = Flickr(root="./data/Flickr")
            data = dataset[0]
        elif self.args.dataset == "Reddit":
            dataset = Reddit(root="./data/Reddit")
            data = dataset[0]
        elif self.args.dataset == "Coauthor-CS":
            dataset = Coauthor(root="./data/Coauthor-CS", name="CS")
            data = dataset[0]
        elif self.args.dataset == "Coauthor-Physics":
            dataset = Coauthor(root="./data/Coauthor-Physics", name="Physics")
            data = dataset[0]
            # Apply PCA for large feature dimensions
            V = torch.pca_lowrank(data.x, q=1000)[2].to(self.device)
            data.x = torch.matmul(data.x.to(self.device), V)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        print("Dataset loaded.")
        
        # Calculate number of nodes
        nodes = data.edge_index.t().numpy()
        nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))
        self.args.num_of_nodes = nodes.max() + 1

        # Handle feature usage
        if self.args.feature_dim > 0:
            feature_using = int(data.x.shape[1] * self.args.feature_dim)
            data.x = data.x[:, :max(1, feature_using)]
            print(f"Using feature dimension: {self.args.feature_dim} ({max(1, feature_using)} features)")
            data.x = F.normalize(data.x.to(self.device), p=1, dim=1)
        else:
            print("Not using node features")
        
        return data
    
    def dealing_with_edges(self, data):
        """Split edges into train/val/test sets"""
        # Get all edges
        edges = data.edge_index.t().numpy()
        
        # Create unique edges (remove duplicates)
        edge_set = set()
        unique_edges = []
        for edge in edges:
            edge_tuple = tuple(sorted(edge))
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                unique_edges.append(edge)
        
        unique_edges = np.array(unique_edges)
        
        # Split edges: 85% train, 5% val, 10% test
        train_edges, temp_edges = train_test_split(unique_edges, test_size=0.15, random_state=42)
        val_edges, test_edges = train_test_split(temp_edges, test_size=0.67, random_state=42)
        
        # Generate negative edges
        def generate_negative_edges(num_edges, num_nodes):
            neg_edges = []
            edge_set = set([tuple(sorted(edge)) for edge in unique_edges])
            
            while len(neg_edges) < num_edges:
                i = np.random.randint(0, num_nodes)
                j = np.random.randint(0, num_nodes)
                if i != j and tuple(sorted([i, j])) not in edge_set:
                    neg_edges.append([i, j])
                    edge_set.add(tuple(sorted([i, j])))
            return np.array(neg_edges)
        
        val_edges_false = generate_negative_edges(len(val_edges), data.x.shape[0])
        test_edges_false = generate_negative_edges(len(test_edges), data.x.shape[0])
        
        # Update data.edge_index to only include training edges
        train_edge_index = torch.tensor(train_edges.T, dtype=torch.long)
        # Make edges bidirectional
        train_edge_index = torch.cat([train_edge_index, train_edge_index.flip(0)], dim=1)
        data.edge_index = train_edge_index
        
        return data, val_edges, val_edges_false, test_edges, test_edges_false
    
    def initialize_model(self, data):
        """Initialize model based on type"""
        self.model = UnifiedGraphModel(
            data.edge_index,
            embedding_dim=self.args.embedding_dim,
            walk_length=20,
            context_size=10,
            rank_bound=self.args.rank,
            walks_per_node=10,
            num_negative_samples=1,
            p=self.args.p,
            q=self.args.q,
            sparse=True,
            num_nodes=self.args.num_of_nodes,
            input_feature_dim=data.x.shape[1] if self.args.feature_dim > 0 else -1,
            model_type=self.args.model_type,
            device=self.device,
        ).to(self.device)

        # Set data for GNN models
        if self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            self.model.set_data(data)
        
        # Initialize transition matrix for GNN models
        if self.args.model_type not in ["deepwalk", "deepwalk_prop"]:
            self.transition_matrix_v = 1e-1 * torch.eye(self.args.rank, self.args.embedding_dim).to(self.device)
            self.transition_matrix_v.requires_grad = False
        else:
            self.transition_matrix_v = None
        
        # Setup optimizers
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.args.lr)
        self.optimizer2 = None
        
        # Setup data loader
        loader = self.model.loader(batch_size=128, shuffle=True, num_workers=4)
        return loader

    def train_epoch(self, epoch, loader):
        """Train one epoch"""
        # Check if we need to switch training phases
        if (self.args.model_type not in ["deepwalk", "deepwalk_prop"] and 
            epoch == self.args.epoch_threshold - 1 and 
            self.optimizer2 is None):
            
            # Initialize second phase training
            self.transition_matrix_v = 1e-1 * (
                torch.eye(self.args.rank, self.args.embedding_dim) + 
                torch.randn(self.args.rank, self.args.embedding_dim) / 1e4
            ).to(self.device)
            self.transition_matrix_v.requires_grad = True
            
            parameters_v = [{"params": [self.transition_matrix_v]}]
            self.optimizer2 = torch.optim.Adam(parameters_v, lr=0.001, weight_decay=1e-4)
            
            # Freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False
        
        # DeepWalk special handling
        if (self.args.model_type in ["deepwalk", "deepwalk_prop"] and 
            epoch == self.args.epoch_threshold and 
            self.optimizer2 is None):
            
            self.model.V.weight += 1e-4 * torch.randn(self.args.rank, self.args.embedding_dim).to(self.device)
            self.model.V.weight.requires_grad = True
            self.model.U.weight.requires_grad = False
            for g in self.optimizer.param_groups:
                g["lr"] = 0.001

        # Training logic
        use_second_optimizer = (
            self.args.model_type not in ["deepwalk", "deepwalk_prop"] and 
            epoch >= self.args.epoch_threshold and 
            self.optimizer2 is not None
        )
        
        self.model.train()
        total_loss = 0
        
        for pos_rw, neg_rw in loader:
            if use_second_optimizer:
                self.optimizer2.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            # Prepare loss arguments based on model type
            if self.args.model_type in ["deepwalk", "deepwalk_prop"]:
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device), epoch=epoch)
            else:
                loss = self.model.loss(
                    pos_rw.to(self.device), 
                    neg_rw.to(self.device), 
                    trans=self.transition_matrix_v
                )
            
            loss.backward()
            
            if use_second_optimizer:
                self.optimizer2.step()
            else:
                self.optimizer.step()
                
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def train_mlp_predictor(self, model_out, train_edge_label_index, train_edge_label, test_edge_label_index):
        """Train MLP predictor for link prediction"""
        predictor = LinkPredictionMLP(model_out.shape[1], hidden_size=256).to(self.device)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(list(predictor.parameters()), lr=5e-3)

        for i in range(100):
            predictor.train()
            optimizer.zero_grad()

            edge_embeddings = model_out[train_edge_label_index]
            combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
            out = predictor(combined)

            loss = criterion(out.view(-1).to(self.device), train_edge_label.float().to(self.device))
            loss.backward()
            optimizer.step()

        predictor.eval()
        edge_embeddings = model_out[test_edge_label_index]
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        out = predictor(combined)

        return out

    def evaluate(self, data, val_edges, val_edges_false, test_edges, test_edges_false, flag=0):
        """Evaluate model"""
        self.model.eval()
        
        # Convert to numpy arrays if they're lists
        val_edges = np.array(val_edges)
        val_edges_false = np.array(val_edges_false) 
        test_edges = np.array(test_edges)
        test_edges_false = np.array(test_edges_false)
        
        # Get model embeddings
        if self.args.model_type == "deepwalk":
            z = self.model(test=1)
        elif self.args.model_type == "deepwalk_prop":
            z = self.model(test=1)
        else:
            z = self.model(trans=self.transition_matrix_v)
        
        model_out = z.detach()

        # Prepare edge data
        train_edge = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=train_edge,
            num_nodes=data.x.shape[0],
            num_neg_samples=train_edge.size(1),
            method="sparse",
        )

        train_edge_label_index = torch.cat([train_edge, neg_edge_index], dim=-1)
        train_edge_label = torch.cat([
            train_edge.new_ones(train_edge.size(1)),
            train_edge.new_zeros(neg_edge_index.size(1)),
        ], dim=0)

        # Validation evaluation
        val_edges_array = val_edges.T
        val_edges_false_array = val_edges_false.T
        val_edge_label_index = torch.cat([
            torch.from_numpy(val_edges_array).to(self.device),
            torch.from_numpy(val_edges_false_array).to(self.device),
        ], dim=-1)
        val_edge_label = torch.cat([
            data.edge_index.new_ones(val_edges_array.shape[1]),
            data.edge_index.new_zeros(val_edges_false_array.shape[1]),
        ], dim=0)

        out = self.train_mlp_predictor(model_out.detach(), train_edge_label_index, 
                                     train_edge_label, val_edge_label_index)
        roc_score_val = roc_auc_score(val_edge_label.float().detach().numpy(), 
                                    out.view(-1).detach().cpu().numpy())
        ap_score_val = average_precision_score(val_edge_label.float().detach().numpy(), 
                                             out.view(-1).detach().cpu().numpy())

        # Test evaluation
        test_edges_array = test_edges.T
        test_edges_false_array = test_edges_false.T
        test_edge_label_index = torch.cat([
            torch.from_numpy(test_edges_array).to(self.device),
            torch.from_numpy(test_edges_false_array).to(self.device),
        ], dim=-1)
        test_edge_label = torch.cat([
            data.edge_index.new_ones(test_edges_array.shape[1]),
            data.edge_index.new_zeros(test_edges_false_array.shape[1]),
        ], dim=0)

        out = self.train_mlp_predictor(model_out.detach(), train_edge_label_index, 
                                     train_edge_label, test_edge_label_index)
        roc_score_test = roc_auc_score(test_edge_label.float().detach().numpy(),
                                     out.view(-1).detach().cpu().numpy())
        ap_score_test = average_precision_score(test_edge_label.float().detach().numpy(),
                                              out.view(-1).detach().cpu().numpy())

        # Calculate rank
        rank = 0.0
        if flag == 1:
            if self.args.cluster == 1 and self.args.dataset not in ["Reddit", "Flickr"]:
                rank = np.linalg.matrix_rank(model_out.detach().cpu().numpy())
            elif self.args.cluster == 0:
                rank = torch.linalg.matrix_rank(model_out.detach())

        return (roc_score_val, ap_score_val, roc_score_test, ap_score_test, rank,
                out.flatten().detach().cpu().numpy())
    
    def build_graph_with_max_nodes(self, edges, max_nodes):
        """Build NetworkX graph with all nodes up to max_nodes"""
        edgelist = [tuple(edge) for edge in edges.cpu().numpy().T]
        G = nx.from_edgelist(edgelist)
        existing_nodes = set(G.nodes())
        missing_nodes = set(range(max_nodes)) - existing_nodes
        for node in missing_nodes:
            G.add_node(node)
        return G

    def eval_hits(self, y_pred_pos, y_pred_neg, K):
        """Calculate hits@K metric"""
        if len(y_pred_neg) < K:
            return 1.0
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
        return hitsK

    def low_degree_metrics(self, preds_all_test, global_graph, test_edges, test_edges_neg):
        """Calculate degree-based metrics"""
        test_edges = np.array(test_edges)
        test_edges_neg = np.array(test_edges_neg)
        preds_all_test = preds_all_test.astype(np.float64)
        preds_all_test_torch = torch.from_numpy(preds_all_test)
        test_edge_pos_num = test_edges.shape[0]

        # Calculate hits
        hits_metrics = {}
        for k in [10, 20, 30, 40, 50]:
            hits_metrics[f"hits{k}"] = self.eval_hits(
                preds_all_test_torch[:test_edge_pos_num],
                preds_all_test_torch[test_edge_pos_num:], k
            )

        # Degree analysis
        preds_all = preds_all_test
        labels_all = np.hstack([np.ones(test_edges.shape[0]), np.zeros(test_edges_neg.shape[0])])
        the_concatenation = np.concatenate((test_edges.flatten(), test_edges_neg.flatten()))
        degrees = np.array([global_graph.degree(node) for node in the_concatenation])
        degree_ranks = np.argsort(np.argsort(degrees))

        # Low degree analysis
        position_marker = degree_ranks < (degree_ranks.shape[0] / 2.0)
        low_degree_nodes = the_concatenation.flatten()[position_marker]
        low_degree_marker = []
        for edge in np.concatenate((test_edges, np.array(test_edges_neg)), axis=0):
            if edge[0] in low_degree_nodes and edge[1] in low_degree_nodes:
                low_degree_marker.append(True)
            else:
                low_degree_marker.append(False)

        roc_score_low_degree = roc_auc_score(labels_all[low_degree_marker], preds_all[low_degree_marker])
        ap_score_low_degree = average_precision_score(labels_all[low_degree_marker], preds_all[low_degree_marker])

        # High degree analysis
        position_marker = degree_ranks > (degree_ranks.shape[0] / 2.0)
        high_degree_nodes = the_concatenation.flatten()[position_marker]
        high_degree_marker = []
        for edge in np.concatenate((test_edges, np.array(test_edges_neg)), axis=0):
            if edge[0] in high_degree_nodes and edge[1] in high_degree_nodes:
                high_degree_marker.append(True)
            else:
                high_degree_marker.append(False)

        roc_score_high_degree = roc_auc_score(labels_all[high_degree_marker], preds_all[high_degree_marker])
        ap_score_high_degree = average_precision_score(labels_all[high_degree_marker], preds_all[high_degree_marker])

        # Add degree-specific hits
        truncation_marker_low = low_degree_marker[:test_edges.shape[0]]
        truncation_marker_high = high_degree_marker[:test_edges.shape[0]]
        
        for k in [10, 20, 30, 40, 50]:
            hits_metrics[f"hits{k}_low_degree"] = self.eval_hits(
                preds_all_test_torch[:test_edge_pos_num][truncation_marker_low],
                preds_all_test_torch[test_edge_pos_num:], k
            )
            hits_metrics[f"hits{k}_high_degree"] = self.eval_hits(
                preds_all_test_torch[:test_edge_pos_num][truncation_marker_high],
                preds_all_test_torch[test_edge_pos_num:], k
            )

        final_metrics = {
            **hits_metrics,
            "roc_score_low_degree": roc_score_low_degree,
            "ap_score_low_degree": ap_score_low_degree,
            "roc_score_high_degree": roc_score_high_degree,
            "ap_score_high_degree": ap_score_high_degree,
        }
        
        return final_metrics

    def save_results(self, roc_score_val, ap_score_val, roc_score_test, ap_score_test, 
                    rank, preds_all_test, test_edges, test_edges_false, hits_and_degree_bias):
        """Save results to files"""
        if self.args.write == 1:
            comp = list(hits_and_degree_bias.values())
            formatted_comp = [f"{num:.4f}" for num in comp]
            comp_string = ",".join(formatted_comp)

            prediction_dict = {
                "predicted": preds_all_test,
                "num_of_positives": len(test_edges),
                "test_edges": test_edges,
                "test_edges_false": test_edges_false,
            }

            # Create results directory
            results_dir = "./example_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Model-specific file naming
            model_filename_map = {
                "deepwalk": "deepwalk_link_pred_completed.txt",
                "deepwalk_prop": "deepwalk_prop_link_pred_completed.txt", 
                "gcn": "gcn_link_pred_completed.txt",
                "sage": "sage_link_pred_completed.txt",
                "gnn_no_prop": "gnn_no_prop_link_pred_completed.txt"
            }
            
            results_file = f"{results_dir}/{model_filename_map[self.args.model_type]}"
            
            # Write header if file doesn't exist
            if not os.path.exists(results_file):
                print(f"{results_file} does not exist, creating it...")
                with open(results_file, "a") as f:
                    formatted_comp_keys = list(hits_and_degree_bias.keys())
                    comp_string_keys = ",".join(formatted_comp_keys)
                    f.write("model,dataset,roc_score_val,ap_score_val,roc_score_test,ap_score_test,"
                           "rank,rank_bound,embedding_dim,p,q,feature_dim,seed," + 
                           comp_string_keys + ",compidx\n")

            # Write results
            with open(results_file, "a") as file:
                line = (f"{self.args.model_type},{self.args.dataset},{roc_score_val:.4f},{ap_score_val:.4f},"
                       f"{roc_score_test:.4f},{ap_score_test:.4f},{rank:.1f},{self.args.rank:.1f},"
                       f"{self.args.embedding_dim:.1f},{self.args.p:.1f},{self.args.q:.1f},"
                       f"{self.args.feature_dim:.8f},{self.args.seed:.1f}," + comp_string + "," + 
                       str(self.args.comp_idx) + "\n")
                file.write(line)

            # Save prediction results with model-specific filenames
            prediction_filename = (f"{results_dir}/{self.args.model_type}_lp_{self.args.dataset}_"
                                 f"{self.args.rank:.0f}_{self.args.embedding_dim:.0f}_{self.args.p:.0f}_"
                                 f"{self.args.q:.0f}_{self.args.feature_dim:.8f}_{self.args.seed:.0f}.npy")
            
            np.save(prediction_filename, prediction_dict)

            print(f"Results saved to {results_file}")
            print(f"Predictions saved to {prediction_filename}")

    def run(self):
        """Run the complete training and evaluation pipeline"""
        # Load data
        data = self.load_dataset()
        
        # Split edges
        data, val_edges, val_edges_false, test_edges, test_edges_false = self.dealing_with_edges(data)
        
        # Initialize model and loader
        loader = self.initialize_model(data)
        
        # Training loop
        print("Starting training...")
        for epoch in tqdm(range(self.args.epochs)):
            loss = self.train_epoch(epoch, loader)
            
            # Optional: periodic evaluation during training
            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

        # Final evaluation
        print("Evaluating model...")
        (roc_score_val, ap_score_val, roc_score_test, ap_score_test, 
         rank, preds_all_test) = self.evaluate(
            data, val_edges, val_edges_false, test_edges, test_edges_false, flag=1
        )

        print(f"Final Results - ROC Val: {roc_score_val:.4f}, AP Val: {ap_score_val:.4f}, "
              f"ROC Test: {roc_score_test:.4f}, AP Test: {ap_score_test:.4f}, Rank: {rank:.4f}")

        # Compute additional metrics
        global_graph = self.build_graph_with_max_nodes(data.edge_index, self.args.num_of_nodes)
        hits_and_degree_bias = self.low_degree_metrics(preds_all_test, global_graph, test_edges, test_edges_false)
        print("Additional metrics:", hits_and_degree_bias)

        # Save results
        self.save_results(roc_score_val, ap_score_val, roc_score_test, ap_score_test,
                         rank, preds_all_test, test_edges, test_edges_false, hits_and_degree_bias)

        return (roc_score_val, ap_score_val, roc_score_test, ap_score_test, rank), hits_and_degree_bias