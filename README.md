# SEESAW

This is the open-source code for *Seesaw: Do Graph Neural Networks Improve Node Representation Learning for All?*



## Environment

We run the corresponding experiments on either NVIDIA P100 or V100 GPUs. Specifically, the machine is configured with 12 virtual CPU cores and 64 GB of RAM for most experiments.

Details can be found in requirements.txt.



## Usage Examples



#### (1) Node classification among seeds and feature dimensions:

```sh
cd scripts
./run_node_classification.sh
```



#### (2) Link prediction among seeds and feature dimensions:

```sh
cd scripts
./run_link_prediction.sh
```



#### (3) Node classification and link prediction w/ & w/o propagation:

```sh
cd scripts
./run_prop_study.sh
```
