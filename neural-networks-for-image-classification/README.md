# Neural networks for image classification

A two-layer NumPy network classifies points on two moons. A PyTorch experiment then compares a linear pixel classifier and a convolutional network on CIFAR-10. The notebook includes gradient checks, decision boundaries, and first-layer filters.

[Open the notebook](neural_networks_for_image_classification.ipynb) · [Per-epoch results](results/recorded-results.json)

## Recorded results

| CIFAR-10 model | Epochs | Correct on 10,000 test images | Accuracy |
|---|---:|---:|---:|
| Linear pixel classifier | 10 | 4,136 | 41.36% |
| Two-convolution network | 20 | 6,508 | 65.08% |

The CNN gained 23.72 percentage points with twice the training epochs. The runs use the same test set and different training budgets.

On the two-moons data, analytical gradients passed a central finite-difference check. Across five ten-epoch runs with seven hidden units, final validation error ranged from **0.6% to 1.8%**. The notebook also shows how the decision boundary changes with hidden width.

## Run

From this directory in a Python 3.11+ environment:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab neural_networks_for_image_classification.ipynb
```

The notebook downloads the two-moons splits, a sample image, and CIFAR-10 into `data/`. The CIFAR-10 section selects CUDA, Apple MPS, or CPU. The NumPy parameter sweeps and convolutional training can take time. Selected plots and epoch summaries are already visible.

## Sources

The two-moons data and filter image come from ENS WILLOW. The experiments draw on materials by Nicolas Le Roux, Andrea Vedaldi, Andrew Zisserman, and Rob Fergus adapted by Gül Varol and Ignacio Rocco. CIFAR-10 is the dataset of Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/practicals/overview/index.html) · [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
