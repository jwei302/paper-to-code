# ResNet-18 Implementation

A PyTorch implementation of the ResNet-18 architecture from the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) (He et al., CVPR 2016).

## Paper Overview

ResNet introduced **residual learning** to address the degradation problem in deep networks. The key insight is that instead of learning a desired mapping H(x) directly, the network learns the residual F(x) = H(x) - x. This is implemented via skip connections that add the input directly to the output of stacked layers.

### Key Contributions
- **Skip connections**: Allow gradients to flow directly through the network, enabling training of very deep networks (100+ layers)
- **Residual blocks**: Building blocks that learn residual functions rather than unreferenced mappings
- **Batch normalization**: Applied after each convolution for stable training

### Architecture

```
Input (224x224x3)
    │
    ▼
[7x7 conv, 64, stride 2] → BatchNorm → ReLU
    │
    ▼
[3x3 max pool, stride 2]
    │
    ▼
[Stage 1: 2x ResBlock, 64 channels]
    │
    ▼
[Stage 2: 2x ResBlock, 128 channels, stride 2]
    │
    ▼
[Stage 3: 2x ResBlock, 256 channels, stride 2]
    │
    ▼
[Stage 4: 2x ResBlock, 512 channels, stride 2]
    │
    ▼
[Global Average Pool]
    │
    ▼
[Fully Connected → n_classes]
```

Each ResBlock contains:
```
x ──┬──► [3x3 conv] → BN → ReLU → [3x3 conv] → BN ──┬──► ReLU → out
    │                                                │
    └────────────────── (projection) ────────────────┘
```

## Usage

### Minimum Working Example

```python
import torch
from resnets import ResNet18, train, evaluate

# Create model for CIFAR-10 (10 classes, 3-channel RGB images)
model = ResNet18(n_input_channels=3, n_classes=10)

# Example forward pass
x = torch.randn(32, 3, 224, 224)  # Batch of 32 images
logits = model(x)  # Output: (32, 10)

# Training
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
model = train(
    model=model,
    dataloader=train_loader,
    n_epochs=90,
    optimizer=optimizer,
    lr_decay=0.1,
    lr_decay_period=30,
    device='cuda'
)

# Evaluation
accuracy = evaluate(
    model=model,
    dataloader=test_loader,
    class_names=['plane', 'car', 'bird', ...],
    device='cuda'
)
```

### Training on CIFAR-10

See `resnets.ipynb` for a complete example training on CIFAR-10.

## Files

- `resnets.py` - Model architecture, training, and evaluation functions
- `utils.py` - Visualization utilities for plotting images and training curves
- `resnets.ipynb` - Jupyter notebook with training example

## Requirements

- PyTorch >= 1.0
- NumPy
- Matplotlib

## References

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}
```
