# SEESAW: Shallow Embedding Methods vs. Graph Neural Networks

This repository contains the implementation for the paper **"SEESAW: Do Graph Neural Networks Improve Node Representation Learning for All?"** published in the Journal of Data-centric Machine Learning Research (2025).

## Overview

SEESAW provides a systematic comparison between shallow graph embedding methods (like DeepWalk) and Graph Neural Networks (GNNs) across multiple datasets and tasks. Our findings reveal that GNNs have notable drawbacks in certain scenarios:

- **Dimensional collapse** in attribute-poor scenarios
- **Performance degradation** on heterophilic nodes due to neighborhood aggregation

## Citation

```bibtex
@article{dong2025seesaw,
  title={SEESAW: Do Graph Neural Networks Improve Node Representation Learning for All?},
  author={Dong, Yushun and Shiao, William and Liu, Yozen and Li, Jundong and Shah, Neil and Zhao, Tong},
  journal={Journal of Data-centric Machine Learning Research},
  year={2025}
}
```


## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

**Node Classification:**
```bash
# Run single experiment
python node_classification.py --model_type deepwalk --dataset Cora --rank 32 --embedding_dim 32

# Run comprehensive experiments
./node_classification.sh
```

**Link Prediction:**
```bash
# Run single experiment  
python link_prediction.py --dataset Cora --rank 32 --embedding_dim 32

# Run comprehensive experiments
./link_prediction.sh
```

## Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--model_type` | Model architecture | `deepwalk`, `gcn`, `sage`, `deepwalk_prop`, `gnn_no_prop` |
| `--dataset` | Graph dataset | `Cora`, `CiteSeer`, `PubMed`, `CoraFull`, `Amazon-Computers`, etc. |
| `--rank` | Rank bound for embedding matrix | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| `--embedding_dim` | Embedding dimension | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| `--feature_dim` | Fraction of features to use | 1.0 (all), 0.1, 0.01, 0.0 (none) |
| `--epochs` | Training epochs | 400 (default for DeepWalk), 200 (for GNNs) |

## Project Structure

```
├── data/
│   └── Cora/
│       └── readme.md
├── example_results/
│   ├── *.txt
│   ├── *.npy
├── link_prediction.py
├── link_prediction.sh
├── node_classification.py
├── node_classification.sh
├── requirements.txt
└── src/
    ├── gnn.py
    ├── link_prediction_training_eval_pipeline.py
    ├── tools.py
    ├── training_pipeline.py
    └── unified_graph_model.py
```

## Model Types

- **`deepwalk`**: Traditional shallow embedding method
- **`deepwalk_prop`**: DeepWalk with propagation enhancement
- **`gcn`**: Graph Convolutional Network
- **`sage`**: GraphSAGE
- **`gnn_no_prop`**: GNN without neighborhood aggregation

## Understanding Results

Results are automatically saved to `example_results/` with comprehensive metrics:

- **Node Classification**: Accuracy, F1-score (macro/micro), recall
- **Link Prediction**: ROC-AUC, Average Precision, Hits@K
- **Additional Analysis**: Performance by node degree, homophily levels

## Key Findings

1. **Attribute-Poor Scenarios**: Shallow methods outperform GNNs when limited node features are available
2. **Heterophilic Networks**: GNNs struggle with nodes having dissimilar neighbors
3. **Dimensional Collapse**: GNNs suffer from representation collapse in low-attribute settings
4. **Recommendation**: Use shallow methods for attribute-poor or heterophilic graphs; GNNs for attribute-rich, homophilic scenarios




## Contact

For questions and support:

* Create an issue on GitHub

* Contact the development team: Yushun Dong (yd24f@fsu.edu).