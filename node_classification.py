import argparse
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
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
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential, SAGEConv, MessagePassing
from torch_geometric.data import Data
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder


from src.gnn import GraphNeuralNetwork
from src.tools import index2ptr, SimpleMessagePassing, MlpProdDecoder
from src.node_classification_training_eval_pipeline import GraphTrainingPipeline
from src.unified_graph_model import UnifiedGraphModel



import warnings
warnings.filterwarnings("ignore")







def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Hardware
    parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA training.")
    
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset: Cora, CiteSeer, PubMed, CoraFull, Amazon-Computers, Amazon-Photo, "
             "CitationFull-Cora, CitationFull-DBLP, Flickr, Reddit, Coauthor-CS, Coauthor-Physics."
    )
    
    # Model selection - NEW PARAMETER
    parser.add_argument(
        "--model_type",
        type=str,
        default="deepwalk",
        choices=["deepwalk", "gcn", "sage", "deepwalk_prop", "gnn_no_prop"],
        help="Model type to use"
    )
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train.")
    parser.add_argument("--p", type=float, default=1.0, help="Node2Vec return parameter.")
    parser.add_argument("--q", type=float, default=1.0, help="Node2Vec in-out parameter.")
    parser.add_argument("--epoch_threshold", type=int, default=300, help="Epoch threshold for training phase transition.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    
    # Model architecture
    parser.add_argument("--rank", type=int, default=32, help="Rank bound for embedding matrix.")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of embeddings.")
    parser.add_argument("--feature_dim", type=float, default=1.0, help="Fraction of features to use.")
    
    # System parameters
    parser.add_argument("--num_of_nodes", type=int, default=-1, help="Number of nodes (auto-detected).")
    parser.add_argument("--write", type=int, default=1, help="Write results to files.")
    parser.add_argument("--cluster", type=int, default=0, help="Cluster mode.")
    parser.add_argument("--comp_idx", type=int, default=-1, help="Component index.")
    parser.add_argument("--verbose", type=bool, default=False, help="Print intermediate results.")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Set feature_dim based on model type if not explicitly set
    if hasattr(args, 'feature_dim'):
        if args.model_type in ["deepwalk", "deepwalk_prop"] and args.feature_dim == 1.0:
            args.feature_dim = 0.0  # DeepWalk variants don't use features by default
    
    # Adjust default parameters based on model type
    if args.model_type in ["gcn", "sage", "gnn_no_prop"]:
        if args.epochs == 400:  # If using default
            args.epochs = 200
        if args.epoch_threshold == 300:  # If using default
            args.epoch_threshold = 100
    
    # Setup device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    print(f"Running {args.model_type} on {args.dataset} with device: {device}")
    print(f"Parameters: rank={args.rank}, embedding_dim={args.embedding_dim}, "
          f"epochs={args.epochs}, epoch_threshold={args.epoch_threshold}")
    
    # Validate arguments
    assert args.embedding_dim >= args.rank, "Embedding dimension must be >= rank"
    
    # Create and run pipeline
    pipeline = GraphTrainingPipeline(args, device)
    results, predictions = pipeline.run()
    
    print("Training and evaluation completed successfully!")