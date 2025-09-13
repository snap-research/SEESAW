from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score
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
import networkx as nx
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

from src.unified_graph_model import UnifiedGraphModel
from src.tools import MlpProdDecoder




class GraphTrainingPipeline:
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
            # Apply PCA for Coauthor-Physics in certain models
            if self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
                V = torch.pca_lowrank(data.x, q=1000)[2].to(self.device)
                data.x = torch.matmul(data.x.to(self.device), V)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        print("Dataset loaded.")
        
        # Add train/val/test masks if not present
        if not hasattr(data, 'train_mask'):
            data = self.dataset_split(data)
            
        assert hasattr(data, 'val_mask'), "Dataset must have validation mask"
        
        self.args.num_of_nodes = data.x.shape[0]
        
        # One-hot encode labels
        enc = OneHotEncoder()
        enc.fit(data.y.view(-1).reshape(-1, 1))
        data.y = enc.transform(data.y.view(-1).reshape(-1, 1)).toarray()
        
        # Feature processing based on model type
        if self.args.model_type == "deepwalk" or self.args.model_type == "deepwalk_prop":
            # DeepWalk variants don't use features
            pass
        else:
            # Use features for GNN models
            feature_using = int(data.x.shape[1] * self.args.feature_dim)
            data.x = data.x[:, :max(1, feature_using)]
            print(f"Using feature dim: {self.args.feature_dim}")
            print(f"Actual features used: {max(1, feature_using)}")
            data.x = F.normalize(data.x.to(self.device), p=1, dim=1)
        
        return data
    
    def dataset_split(self, data):
        """Split dataset into train/val/test"""
        nodes = data.edge_index.t().numpy()
        nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))
        total_num = nodes.max() + 1
        np.random.shuffle(nodes)

        # 60% - 20% - 20% split
        train_size = int(total_num * 0.6)
        test_size = int(total_num * 0.8) - train_size
        val_size = total_num - train_size - test_size

        train_set = nodes[0:train_size]
        test_set = nodes[train_size:train_size + test_size]
        val_set = nodes[train_size + test_size:]

        train_mask = torch.zeros(total_num, dtype=torch.bool, device=self.device)
        for i in train_set:
            train_mask[i] = True

        test_mask = torch.zeros(total_num, dtype=torch.bool, device=self.device)
        for i in test_set:
            test_mask[i] = True

        val_mask = torch.zeros(total_num, dtype=torch.bool, device=self.device)
        for i in val_set:
            val_mask[i] = True

        data.train_mask = train_mask.cpu().numpy()
        data.test_mask = test_mask.cpu().numpy()
        data.val_mask = val_mask.cpu().numpy()

        return data
    
    def initialize_model(self, data):
        """Initialize model based on type"""
        input_feature_dim = data.x.shape[1] if hasattr(data, 'x') and data.x is not None else 0
        
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
            input_feature_dim=input_feature_dim,
            model_type=self.args.model_type,
            device=self.device,
        ).to(self.device)
        
        # Set data for GNN models
        if self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            self.model.set_data(data)
        
        # Initialize transition matrix for GNN models
        if self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            self.transition_matrix_v = 1e-1 * torch.eye(self.args.rank, self.args.embedding_dim).to(self.device)
            self.transition_matrix_v.requires_grad = False
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.args.lr)
        
        return self.model.loader(batch_size=128, shuffle=True, num_workers=4)
    
    def train_epoch(self, epoch, loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        # Determine training phase
        if self.args.model_type == "deepwalk":
            # DeepWalk training
            for pos_rw, neg_rw in loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device), epoch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
        elif self.args.model_type == "deepwalk_prop":
            # DeepWalk with propagation training
            for pos_rw, neg_rw in loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device), epoch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
        elif self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            # GNN training
            if epoch < self.args.epoch_threshold:
                for pos_rw, neg_rw in loader:
                    self.optimizer.zero_grad()
                    loss = self.model.loss(
                        pos_rw.to(self.device), neg_rw.to(self.device), 
                        trans=self.transition_matrix_v
                    )
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            else:
                for pos_rw, neg_rw in loader:
                    self.optimizer2.zero_grad()
                    loss = self.model.loss(
                        pos_rw.to(self.device), neg_rw.to(self.device), 
                        trans=self.transition_matrix_v
                    )
                    loss.backward()
                    self.optimizer2.step()
                    total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def handle_epoch_threshold(self, epoch):
        """Handle epoch threshold transitions"""
        if self.args.model_type == "deepwalk" and epoch == self.args.epoch_threshold:
            self.model.V.weight += 1e-4 * torch.randn(self.args.rank, self.args.embedding_dim).to(self.device)
            self.model.V.weight.requires_grad = True
            self.model.U.weight.requires_grad = False
            for g in self.optimizer.param_groups:
                g["lr"] = 0.001
                
        elif self.args.model_type == "deepwalk_prop" and epoch == self.args.epoch_threshold:
            self.model.V.weight += 1e-4 * torch.randn(self.args.rank, self.args.embedding_dim).to(self.device)
            self.model.V.weight.requires_grad = True
            self.model.U.weight.requires_grad = False
            for g in self.optimizer.param_groups:
                g["lr"] = 0.001
                
        elif self.args.model_type in ["gcn", "sage", "gnn_no_prop"] and epoch == self.args.epoch_threshold - 1:
            self.transition_matrix_v = 1e-1 * (
                torch.eye(self.args.rank, self.args.embedding_dim) + 
                torch.randn(self.args.rank, self.args.embedding_dim) / 1e4
            ).to(self.device)
            self.transition_matrix_v.requires_grad = True
            
            parameters_v = [{"params": [self.transition_matrix_v]}]
            self.optimizer2 = torch.optim.Adam(parameters_v, lr=0.001, weight_decay=1e-4)
            
            for param in self.model.parameters():
                param.requires_grad = False
    
    def train_classifier(self, model_out, data, class_num):
        """Train MLP classifier on embeddings"""
        predictor = MlpProdDecoder(
            self.args.embedding_dim, hidden_size=256, class_num=class_num
        ).to(self.device)
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(list(predictor.parameters()), lr=5e-3)

        train_labels = torch.from_numpy(data.y[data.train_mask]).to(self.device)

        for i in range(100):
            predictor.train()
            optimizer.zero_grad()
            edge_embeddings = model_out[data.train_mask]
            out = predictor(edge_embeddings)
            loss = criterion(out.to(self.device), train_labels.float())
            loss.backward()
            optimizer.step()

        predictor.eval()
        edge_embeddings = model_out
        out = predictor(edge_embeddings)
        return out
    
    def evaluate(self, data, flag=0):
        """Evaluate model"""
        self.model.eval()
        
        # Get embeddings based on model type
        if self.args.model_type in ["deepwalk", "deepwalk_prop"]:
            z = self.model(test=1)
        elif self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            z = self.model(trans=self.transition_matrix_v)
        
        model_out = z.detach()
        
        # Train classifier and get predictions
        out = self.train_classifier(model_out.detach(), data, class_num=data.y.shape[1])
        out_cpu = out.detach().cpu().numpy()
        
        # Process outputs
        out_cpu_val = np.argmax(out_cpu[data.val_mask], axis=1)
        out_cpu_test = np.argmax(out_cpu[data.test_mask], axis=1)
        out_cpu_val_multi_dim = softmax(out_cpu[data.val_mask], axis=1)
        out_cpu_test_multi_dim = softmax(out_cpu[data.test_mask], axis=1)
        
        val_labels = np.argmax(data.y[data.val_mask], axis=1)
        test_labels = np.argmax(data.y[data.test_mask], axis=1)
        
        # Validation metrics
        accuracy_score_val = accuracy_score(val_labels, out_cpu_val)
        macro_f1_val = f1_score(val_labels, out_cpu_val, average="macro")
        micro_f1_val = f1_score(val_labels, out_cpu_val, average="micro")
        recall_macro_val = recall_score(val_labels, out_cpu_val, average="macro")
        recall_micro_val = recall_score(val_labels, out_cpu_val, average="micro")

        # Test metrics
        accuracy_score_test = accuracy_score(test_labels, out_cpu_test)
        macro_f1_test = f1_score(test_labels, out_cpu_test, average="macro")
        micro_f1_test = f1_score(test_labels, out_cpu_test, average="micro")
        recall_macro_test = recall_score(test_labels, out_cpu_test, average="macro")
        recall_micro_test = recall_score(test_labels, out_cpu_test, average="micro")

        # Matrix rank calculation
        rank = 0.000
        if flag == 1:
            if self.args.cluster == 1 and self.args.dataset not in ["Reddit", "Flickr"]:
                rank = np.linalg.matrix_rank(model_out.detach().cpu().numpy())
            elif self.args.cluster == 0:
                rank = torch.linalg.matrix_rank(model_out.detach())

        return_dict = {}
        return_dict["accuracy_score_val"] = accuracy_score_val
        return_dict["macro_f1_val"] = macro_f1_val
        return_dict["micro_f1_val"] = micro_f1_val
        return_dict["recall_macro_val"] = recall_macro_val
        return_dict["recall_micro_val"] = recall_micro_val

        return_dict["accuracy_score_test"] = accuracy_score_test
        return_dict["macro_f1_test"] = macro_f1_test
        return_dict["micro_f1_test"] = micro_f1_test
        return_dict["recall_macro_test"] = recall_macro_test
        return_dict["recall_micro_test"] = recall_micro_test

        return_dict["rank"] = rank

        # Degree analysis
        global_graph = self.build_graph_with_max_nodes(data.edge_index, self.args.num_of_nodes)
        the_concatenation = np.arange(self.args.num_of_nodes)[data.test_mask]
        degrees = np.array([global_graph.degree(node) for node in the_concatenation])
        degree_ranks = np.argsort(np.argsort(degrees))

        low_degree_nodes_marker = degree_ranks < (degree_ranks.shape[0] / 2.0)
        high_degree_nodes_marker = degree_ranks > (degree_ranks.shape[0] / 2.0)

        return_dict["accuracy_score_test_low_degree_nodes"] = accuracy_score(
            test_labels[low_degree_nodes_marker], out_cpu_test[low_degree_nodes_marker]
        )
        return_dict["macro_f1_test_low_degree_nodes"] = f1_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="macro",
        )
        return_dict["micro_f1_test_low_degree_nodes"] = f1_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="micro",
        )
        return_dict["recall_macro_test_low_degree_nodes"] = recall_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="macro",
        )
        return_dict["recall_micro_test_low_degree_nodes"] = recall_score(
            test_labels[low_degree_nodes_marker],
            out_cpu_test[low_degree_nodes_marker],
            average="micro",
        )

        return_dict["accuracy_score_test_high_degree_nodes"] = accuracy_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
        )
        return_dict["macro_f1_test_high_degree_nodes"] = f1_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="macro",
        )
        return_dict["micro_f1_test_high_degree_nodes"] = f1_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="micro",
        )
        return_dict["recall_macro_test_high_degree_nodes"] = recall_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="macro",
        )
        return_dict["recall_micro_test_high_degree_nodes"] = recall_score(
            test_labels[high_degree_nodes_marker],
            out_cpu_test[high_degree_nodes_marker],
            average="micro",
        )

        return return_dict, out_cpu[data.test_mask]
    
    def build_graph_with_max_nodes(self, edges, max_nodes):
        """Build NetworkX graph with all nodes up to max_nodes"""
        edgelist = [tuple(edge) for edge in edges.cpu().numpy().T]
        G = nx.from_edgelist(edgelist)
        num_nodes = G.number_of_nodes()
        existing_nodes = set(G.nodes())
        missing_nodes = set(range(max_nodes)) - existing_nodes

        for node in missing_nodes:
            G.add_node(node)

        return G
    
    def save_results(self, return_dict, predictions, data):
        """Save results to files"""
        if self.args.write == 1:
            # Prepare data for saving
            comp = list(return_dict.values())
            formatted_comp = [f"{num:.4f}" for num in comp]
            comp_string = ",".join(formatted_comp)
            
            prediction_dict = {
                "predicted": predictions,
                "num_of_positives": data.y[data.test_mask],
                "test_mask": np.arange(self.args.num_of_nodes)[data.test_mask],
            }
            

            # if self.args.model_type == "deepwalk_prop":
            #     file_prefix = "deepwalk_prop_node_classification_completed.txt"
            #     result_prefix = "deepwalk_prop"
            # elif self.args.model_type in ["gcn", "sage", "gnn_no_prop"]:
            #     if self.args.model_type == "gnn_no_prop":
            #         file_prefix = "2_node_classification_completed.txt"
            #     else:
            #         file_prefix = "1_node_classification_completed.txt"
            #     result_prefix = self.args.model_type
            # else:  # deepwalk
            #     file_prefix = "1_node_classification_completed.txt"
            #     result_prefix = "deepwalk"

            result_prefix = self.args.model_type
            file_prefix = self.args.model_type + "_node_classification_completed.txt"

            
            
            # Create results directory if it doesn't exist
            os.makedirs("./example_results", exist_ok=True)
            
            results_file = f"./example_results/{file_prefix}"
            
            # Write header if file doesn't exist
            if not os.path.exists(results_file):
                print(f"{file_prefix} does not exist, creating it...")
                with open(results_file, "a") as f:
                    formatted_comp_keys = list(return_dict.keys())
                    comp_string_keys = ",".join(formatted_comp_keys)
                    f.write(
                        "model,dataset,rank_bound,embedding_dim,p,q,feature_dim,seed,"
                        + comp_string_keys
                        + "\n"
                    )

            # Write results
            with open(results_file, "a") as file:
                line = (
                    f"{result_prefix},{self.args.dataset},{self.args.rank:.1f},"
                    f"{self.args.embedding_dim:.1f},{self.args.p:.1f},{self.args.q:.1f},"
                    f"{self.args.feature_dim:.8f},{self.args.seed:.1f},"
                    + comp_string
                    + "\n"
                )
                file.write(line)

            # Save predictions
            prediction_filename = (
                f"./example_results/{result_prefix}_nc_{self.args.dataset}_"
                f"{self.args.rank:.0f}_{self.args.embedding_dim:.0f}_"
                f"{self.args.p:.0f}_{self.args.q:.0f}_{self.args.feature_dim:.8f}_"
                f"{self.args.seed:.0f}.npy"
            )
            np.save(prediction_filename, prediction_dict)
            
            print(f"Results saved to {results_file}")
            print(f"Predictions saved to {prediction_filename}")
    
    def run(self):
        """Run the complete training and evaluation pipeline"""
        # Load data
        data = self.load_dataset()
        
        # Initialize model and loader
        loader = self.initialize_model(data)
        
        # Training loop
        for epoch in tqdm(range(0, self.args.epochs)):
            loss = self.train_epoch(epoch, loader)
            
            # Handle epoch threshold transitions
            self.handle_epoch_threshold(epoch)
            
            # Optional intermediate evaluation for debugging
            if hasattr(self.args, 'verbose') and self.args.verbose and epoch % 5 == 0:
                return_dict, _ = self.evaluate(data, flag=1)
                print(f'Epoch {epoch}: acc_test: {return_dict["accuracy_score_test"]:.4f}, '
                      f'macro_f1_test: {return_dict["macro_f1_test"]:.4f}, '
                      f'rank: {return_dict["rank"]:.4f}')
        
        # Final evaluation
        return_dict, preds_all_test = self.evaluate(data, flag=1)
        print("Final Results:")
        print(return_dict)
        
        # Save results
        self.save_results(return_dict, preds_all_test, data)
        
        return return_dict, preds_all_test
