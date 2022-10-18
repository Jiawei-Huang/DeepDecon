# DeepDecon: A Deep-learning Method for Estimating Cell Fractions in Bulk RNA-seq Data with Applications to AML

### Overview
Here, we present `DeepDecon`, a deep neural network model leveraging single-cell gene expression information to accurately predict the fraction of cancer cells in bulk tissues. `DeepDecon` was trained based on single-cell RNA sequencing data and was robust to experimental biases and noises. It will automatically select optimal models to recursively estimate malignant cell fractions and improve prediction accuracy. When applied to bone marrow data (see Tutorials), it outperforms existing decomposition methods in both accuracy and robustness. We further show that the `DeepDecon` is robust to the number of single cells within a bulk sample.

### Requirements
- tensorflow                1.14.0
- scikit-learn              0.24.2
- python                    3.6.12
- pandas                    1.1.3
- numpy                     1.19.2
- keras                     2.3.1
- scanpy                    1.7.2

### Installation
Download DeepDecon by

```git
git clone https://github.com/Jiawei-Huang/DeepDecon.git
```
Installation has been tested in a Linux and MacOs platform with Python3.6. GPU is recommended for accelerating the training process.

### Instructions
This section provides instructions on how to run scDEC with scRNA-seq datasets.



### Contact
Feel free to open an issue on Github or contact [me](jiaweih@usc.edu) if you have any problem in running DeepDecon.

