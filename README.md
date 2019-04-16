# Dataset REPresentAtion bIas Removal (REPAIR)
This repository contains supplemetary code for paper *REPAIR: Removing Representation Bias by Dataset Resampling* [1]. 

<!-- ## Overview -->

#### What is representation bias?
Representation bias [2] captures the bias of a dataset, in that it is easier to solve using some data representations than others. If a representation is particularly useful for solving dataset D, we say that D is biased towards this representation.

#### What is REPAIR? 
*REPAIR* is a dataset resampling algorithm proposed to reduce *representation bias* of datasets. It learns a set of example-level weights for the dataset that minimizes the bias under reweighting. Based on these weights we obtain the resampled dataset, hopefully with reduced bias, by keeping only a subset of examples (discarding the rest).

#### Why remove the bias?
Neural network models learned on REPAIRed datasets (with bias removed) are shown to generalize better, under domains where such bias is not present. We hypothesize that this is because neural nets trained on biased data will rely on such bias to classify examples, thereby "overfitting" to the bias.

---

### Supported features
- Generating colored MNIST data by **coloring** handwritten digits of MNIST
- **Measure** color bias of the colored MNIST dataset
- **Remove** color bias of dataset with REPAIR resampling
- Evaluate **generalization** of models trained on colored MNIST

### Features to come
- Static bias measurement and removal on **action recognition** datasets: UCF101, HMDB51, Kinetics
- REPAIRed action recognition datasets and **pre-trained models** on them
- **Generic REPAIR tool** for any dataset and representation bias

---

## Instructions
The codes are tested on PyTorch 0.4.1+ only. Two executable scripts are included:
- `colored_mnist.py` creates a Colored MNIST dataset with a provided intra-class color variation, measures its color bias, and evaluates the generalization of a LeNet-5 model trained on the colored dataset.
- `colored_mnist_repair.py` learns an example-level reweighting on the Colored MNIST dataset, performs REPAIR resampling accordingly, and measures the bias and generalization before and after resampling.

---

### References

[1] Yi Li and Nuno Vasconcelos. *REPAIR: Removing Representation Bias by Dataset Resampling.* To appear at CVPR 2019.

[2] Yingwei Li, Yi Li and Nuno Vasconcelos. *RESOUND: Towards Action Recognition without Representation Bias.* ECCV 2018.