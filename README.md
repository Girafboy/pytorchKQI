## Overview

pytorchKQI provides a KQI (Knowledge Quantification Index) calculation API for neural networks based on the PyTorch framework, which can efficiently compute the KQI for various neural networks. The functionalities provided by pytorchKQI include the following three items:
1. Calculation of KQI for neural networks;
2. Calculation of KQI for neural network nodes; 
3. Visualization of KQI for neural networks.

## Installation

#### Install from Github

```bash
git clone https://github.com/Girafboy/pytorchKQI.git
cd pytorchKQI
```

#### Graphviz Dependencies

```bash
sudo apt-get install graphviz graphviz-dev
```

#### Python Dependencies

```
pytest
tqdm
numpy
pandas
networkx
matplotlib
torch==1.12.0
torchaudio==0.12.0
torchvision==0.13.0
transformers
pygraphviz
```

## Running Tests

To run the test, execute the code:

```bash
pytest
```

## Quick tour
To immediately calculate the KQI on a given neural network model, we offer the `torchKQI` API. Here is an example of how to quickly use `torchKQI` to compute the KQI for the AlexNet model.
```python
import torch
import torchvision
import torchKQI

x = torch.randn(1, 3, 224, 224)
model = torchvision.models.alexnet()

# Calculation of KQI for neural networks
kqi = torchKQI.KQI(model, x)
print(kqi)

# Calculation of KQI for neural network nodes
for node_result in torchKQI.Graph(model, x):
    print(node_result)

# Visualization of KQI for neural networks
torchKQI.VisualKQI(model, x)
```

## How to Contribute

Here's how you can contribute to our GitHub repository:

**1.Fork the Repository**
+ On the repository page, click the "Fork" button in the upper right corner to create a copy of the repository under your GitHub account.

**2.Clone the Repository**
+ Clone your forked repository to your local machine:

```bash
git clone https://github.com/{your-username}/pytorchKQI.git
```

**3.Create a Branch**
+ Navigate to the repository directory and create a new branch for your changes.
```bash
cd pytorchKQI
git checkout -b new-feature-branch
```

**4.Make Changes**
+ Work on your branch to make code changes or add new features, following the coding style and guidelines of the project.

**5.Commit Changes**
+ Stage your changes and commit them to your local repository.
```bash
git add .
git commit -m "Add a concise description of your changes"
```

**6.Push Changes**
+ Push your changes to your forked repository on GitHub.

```bash
git push origin new-feature-branch
```

**7.Create a Pull Request**
+ Go to your forked repository page on GitHub and click the "Pull Request" button next to the branch you just pushed.
+ Fill in the title and description of the pull request, explaining the changes you've made.
+ Submit the pull request.
