import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import warnings

# Import the pipeline
from src.link_prediction_training_eval_pipeline import LinkPredictionTrainingPipeline

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
        default="Flickr",
        help="Dataset name: Cora, CiteSeer, PubMed, CoraFull, Amazon-Computers, "
             "Amazon-Photo, CitationFull-DBLP, Flickr, Reddit, Coauthor-CS, Coauthor-Physics."
    )
    
    # Model selection
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="deepwalk", 
        choices=["deepwalk", "deepwalk_prop", "gcn", "sage", "gnn_no_prop"],
        help="Model type to use"
    )
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to train.")
    parser.add_argument("--p", type=float, default=1.0, help="Random walk return parameter.")
    parser.add_argument("--q", type=float, default=1.0, help="Random walk in-out parameter.")
    parser.add_argument("--epoch_threshold", type=int, default=300, help="Epoch to switch training phase.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    
    # Model architecture
    parser.add_argument("--rank", type=int, default=32, help="Rank bound for embedding matrix.")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of embeddings.")
    parser.add_argument("--feature_dim", type=float, default=0.0, 
                       help="Fraction of features to use (0.0 for no features, 1.0 for all).")
    
    # System parameters
    parser.add_argument("--num_of_nodes", type=int, default=-1, help="Number of nodes (auto-detected).")
    parser.add_argument("--write", type=int, default=1, help="Write results to files.")
    parser.add_argument("--cluster", type=int, default=0, help="Cluster mode.")
    parser.add_argument("--comp_idx", type=int, default=-1, help="Computation index.")
    
    return parser.parse_args()



def print_results(results, args):
    """Print final results"""
    roc_score_val, ap_score_val, roc_score_test, ap_score_test, rank = results


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    
    # Setup device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    
    assert args.embedding_dim >= args.rank
    

    print(f"Running {args.model_type} on {args.dataset} with device: {device}")
    print(f"Parameters: rank={args.rank}, embedding_dim={args.embedding_dim}, "
          f"epochs={args.epochs}, epoch_threshold={args.epoch_threshold}")


    # Initialize and run the training pipeline
    pipeline = LinkPredictionTrainingPipeline(args, device)
    results, additional_metrics = pipeline.run()
    
    # Print results
    print_results(results, args)
    
    print("Training and evaluation completed successfully!")