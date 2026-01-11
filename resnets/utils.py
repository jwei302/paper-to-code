"""
Visualization utilities for image classification tasks.

This module provides helper functions for displaying images and predictions
in a grid layout using matplotlib.
"""

from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_images(
    images: List[List[np.ndarray]],
    titles: Optional[List[List[str]]] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    cmap: Optional[str] = None,
    title_fontsize: int = 6,
    show: bool = True
) -> plt.Figure:
    """
    Display images in a grid layout.

    Args:
        images: 2D list of images where images[row][col] is a numpy array.
                Images can be grayscale (H, W) or RGB (H, W, 3).
        titles: Optional 2D list of titles corresponding to each image.
                If None, no titles are displayed.
        figsize: Optional figure size as (width, height) in inches.
                 If None, auto-calculated based on grid size.
        dpi: Dots per inch for the figure (default: 150).
        cmap: Colormap for grayscale images (default: None, uses 'gray' for 2D arrays).
        title_fontsize: Font size for subplot titles (default: 6).
        show: Whether to call plt.show() (default: True).

    Returns:
        The matplotlib Figure object.

    Example:
        >>> images = [[img1, img2], [img3, img4]]
        >>> titles = [['Cat', 'Dog'], ['Bird', 'Fish']]
        >>> plot_images(images, titles)
    """
    n_rows = len(images)
    n_cols = max(len(row) for row in images) if images else 0

    if n_rows == 0 or n_cols == 0:
        raise ValueError("Images list cannot be empty")

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    # Ensure axes is always 2D for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            # Check if image exists at this position
            if col_idx < len(images[row_idx]):
                img = images[row_idx][col_idx]

                # Determine colormap based on image dimensions
                img_cmap = cmap
                if img_cmap is None and img.ndim == 2:
                    img_cmap = 'gray'

                ax.imshow(img, cmap=img_cmap)

                # Set title if provided
                if titles is not None and row_idx < len(titles) and col_idx < len(titles[row_idx]):
                    ax.set_title(titles[row_idx][col_idx], fontsize=title_fontsize)

            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    figsize: tuple = (10, 4),
    show: bool = True
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: Optional list of validation losses per epoch.
        train_accs: Optional list of training accuracies per epoch.
        val_accs: Optional list of validation accuracies per epoch.
        figsize: Figure size as (width, height) in inches.
        show: Whether to call plt.show() (default: True).

    Returns:
        The matplotlib Figure object.

    Example:
        >>> train_losses = [0.9, 0.5, 0.3, 0.2]
        >>> val_losses = [1.0, 0.6, 0.4, 0.35]
        >>> plot_training_curves(train_losses, val_losses)
    """
    n_plots = 1 + (train_accs is not None or val_accs is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot (if provided)
    if n_plots > 1:
        if train_accs is not None:
            axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
        if val_accs is not None:
            axes[1].plot(epochs, val_accs, 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig
