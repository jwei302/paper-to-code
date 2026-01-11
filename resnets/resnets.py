"""
ResNet Implementation

This module implements the ResNet-18 architecture from the paper:
"Deep Residual Learning for Image Recognition" (He et al., 2015)
https://arxiv.org/abs/1512.03385

ResNet introduces skip connections (residual connections) that allow gradients
to flow directly through the network, enabling training of very deep networks.
The key insight is learning residual functions F(x) = H(x) - x rather than
directly learning the desired mapping H(x).

Architecture Overview (ResNet-18):
    - Initial 7x7 conv with stride 2, followed by max pooling
    - 4 stages with [2, 2, 2, 2] residual blocks
    - Channel progression: 64 -> 128 -> 256 -> 512
    - Global average pooling followed by fully connected layer

Example:
    >>> model = ResNet18(n_input_channels=3, n_classes=10)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)  # Shape: (1, 10)
"""

import torch
import torch.nn as nn
from utils import plot_images


class ResNetBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Implements the residual learning formulation:
        output = ReLU(F(x) + x)

    where F(x) consists of two 3x3 convolutions with batch normalization.
    When dimensions change (stride > 1 or channel mismatch), a projection
    shortcut is used to match dimensions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)

    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H', W')
          where H' = H // stride, W' = W // stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # First conv: may downsample spatially via stride
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv: preserves spatial dimensions
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut for dimension matching
        # Used when stride > 1 (spatial downsampling) or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, out_channels, H', W')
        """
        identity = self.projection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection: add identity mapping
        out = out + identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet-18 architecture for image classification.

    The network consists of:
        1. Initial convolution + batch norm + max pool
        2. Four stages of residual blocks (2 blocks each)
        3. Global average pooling
        4. Fully connected classification head

    Stage layout:
        - Stage 1: 64 channels, stride 1
        - Stage 2: 128 channels, stride 2 (downsamples)
        - Stage 3: 256 channels, stride 2 (downsamples)
        - Stage 4: 512 channels, stride 2 (downsamples)

    Args:
        n_input_channels: Number of input image channels (e.g., 3 for RGB)
        n_classes: Number of output classes

    Example:
        >>> model = ResNet18(n_input_channels=3, n_classes=1000)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> logits = model(x)  # Shape: (32, 1000)
    """

    def __init__(self, n_input_channels: int, n_classes: int):
        super().__init__()

        # Initial convolution: 7x7, stride 2, reduces spatial dims by half
        self.conv1 = nn.Conv2d(
            n_input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling: reduces spatial dims by half
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_stage(in_channels=64, out_channels=64, n_blocks=2, stride=1)
        self.layer2 = self._make_stage(in_channels=64, out_channels=128, n_blocks=2, stride=2)
        self.layer3 = self._make_stage(in_channels=128, out_channels=256, n_blocks=2, stride=2)
        self.layer4 = self._make_stage(in_channels=256, out_channels=512, n_blocks=2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """
        Create a stage of residual blocks.

        Args:
            in_channels: Input channels for first block
            out_channels: Output channels for all blocks
            n_blocks: Number of residual blocks in this stage
            stride: Stride for first block (subsequent blocks use stride=1)

        Returns:
            Sequential container of residual blocks
        """
        # First block handles stride and channel changes
        # Subsequent blocks maintain dimensions
        strides = [stride] + [1] * (n_blocks - 1)

        layers = []
        current_channels = in_channels
        for s in strides:
            layers.append(ResNetBlock(current_channels, out_channels, stride=s))
            current_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Classification logits of shape (N, n_classes)
        """
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    lr_decay: float = 1.0,
    lr_decay_period: int = 10,
    device: str = 'cpu'
) -> nn.Module:
    """
    Train a classification model with learning rate scheduling.

    Args:
        model: Neural network to train
        dataloader: DataLoader providing (images, labels) batches
        n_epochs: Number of training epochs
        optimizer: Optimizer for weight updates
        lr_decay: Multiplicative factor for learning rate decay (default: 1.0, no decay)
        lr_decay_period: Apply lr_decay every N epochs (default: 10)
        device: Device to train on ('cpu', 'cuda', 'mps')

    Returns:
        Trained model
    """
    device = torch.device(device)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        # Learning rate scheduling
        if epoch > 0 and epoch % lr_decay_period == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{n_epochs} - Loss: {mean_loss:.4f}')

    return model


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = 'cpu',
    visualize: bool = True,
    grid_size: int = 5
) -> float:
    """
    Evaluate a classification model and optionally visualize predictions.

    Args:
        model: Neural network to evaluate
        dataloader: DataLoader providing (images, labels) batches
        class_names: List mapping class indices to names
        device: Device to run on ('cpu', 'cuda', 'mps')
        visualize: Whether to display prediction grid (default: True)
        grid_size: Size of visualization grid (default: 5x5)

    Returns:
        Classification accuracy as a float between 0 and 1
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    n_correct = 0
    n_samples = 0
    last_batch = None

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

            last_batch = (images, labels, outputs)

    accuracy = n_correct / n_samples
    print(f'Accuracy: {accuracy * 100:.2f}% ({n_correct}/{n_samples})')

    # Visualization of last batch predictions
    if visualize and last_batch is not None:
        images, labels, outputs = last_batch

        # Convert to numpy for plotting
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        labels = labels.cpu().numpy()
        predictions = outputs.argmax(dim=1).cpu().numpy()

        # Create grid of images with titles
        n_display = min(grid_size * grid_size, len(images))
        images_grid = []
        titles_grid = []

        for row in range(grid_size):
            row_images = []
            row_titles = []
            for col in range(grid_size):
                idx = row * grid_size + col
                if idx < n_display:
                    row_images.append(images[idx])
                    pred_name = class_names[predictions[idx]]
                    true_name = class_names[labels[idx]]
                    row_titles.append(f'pred: {pred_name}\ntrue: {true_name}')
            if row_images:
                images_grid.append(row_images)
                titles_grid.append(row_titles)

        if images_grid:
            plot_images(images_grid, titles_grid)

    return accuracy
