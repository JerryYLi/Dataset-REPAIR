# Dataset REPresentAtion bIas Removal (REPAIR)
This repository contains supplemetary code for paper *REPAIR: Removing Representation Bias by Dataset Resampling* [1]. 

### Overview
*REPAIR* is a dataset resampling algorithm proposed to reduce *representation bias* of datasets. It learns a set of example-level of weights for the dataset that minimizes the reweighted bias (?). Neural network models learned on resampled datasets are shown to generalize better (?) under two (scenarios): ...

---

### Supported features
- Generating colored MNIST data by **coloring** handwritten digits of MNIST
- **Measure** color bias of the colored MNIST dataset
- **Remove** color bias of dataset with REPAIR resampling
- Evaluate **generalization** of models trained on colored MNIST

### Features to come
- Comparison between resampling strategies
- Static bias measurement and removal in **action recognition** datasets: UCF101, HMDB51, Kinetics
- REPAIRed datasets and pre-trained models on them
- **Generic REPAIR tool** for any dataset and representation bias

---

## Instructions
The codes are tested on PyTorch 0.4.1+ only. Two executable scripts are included:
- `colored_mnist.py` creates a Colored MNIST dataset with a provided intra-class color variation, measures its color bias, and evaluates the generalization of a LeNet-5 model trained on the colored dataset.
- `colored_mnist_repair.py` learns an example-level reweighting on the Colored MNIST dataset, performs REPAIR resampling accordingly, and measures the bias and generalization before and after resampling.

---

### References

[1] Y. Li and N. Vasconcelos. REPAIR: Removing Representation Bias by Dataset Resampling. To appear at CVPR 2019.
