"""
GPT-2 Transformer Implementation

This module implements the GPT-2 architecture from the paper:
"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

GPT-2 is a decoder-only transformer that uses causal (autoregressive) attention
to generate text. Key architectural choices:
    - Pre-normalization: Layer norm applied before attention/MLP (not after)
    - GELU activation: Smoother alternative to ReLU in feed-forward layers
    - Weight tying: Token embedding weights shared with output projection
    - Learned positional embeddings: Instead of sinusoidal encodings

Architecture Overview (GPT-2 base, 124M params):
    - 12 transformer blocks
    - 12 attention heads per block
    - 768 embedding dimension
    - 1024 context window
    - 50,257 vocabulary (BPE tokens)

Example:
    >>> config = GPTConfig()
    >>> model = GPT(config)
    >>> tokens = torch.randint(0, 50257, (1, 32))
    >>> logits, loss = model(tokens)  # Shape: (1, 32, 50257)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """
    Configuration for GPT-2 model.

    Attributes:
        context_size: Maximum sequence length (default: 1024)
        vocab_size: Vocabulary size (default: 50257 for GPT-2 BPE)
        n_layer: Number of transformer blocks (default: 12)
        n_head: Number of attention heads (default: 12)
        n_embd: Embedding dimension (default: 768)

    Model sizes:
        - gpt2:        n_layer=12, n_head=12, n_embd=768  (124M params)
        - gpt2-medium: n_layer=24, n_head=16, n_embd=1024 (350M params)
        - gpt2-large:  n_layer=36, n_head=20, n_embd=1280 (774M params)
        - gpt2-xl:     n_layer=48, n_head=25, n_embd=1600 (1.5B params)
    """
    context_size: int = 1024
    vocab_size: int = 50257  # 50,000 BPE merges + 256 bytes + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Implements the attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017) with causal masking to prevent attending to
    future tokens.

    The attention computation:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Causal masking ensures position i can only attend to positions <= i.

    Args:
        config: GPTConfig with model hyperparameters
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = True  # Flag for scaled initialization

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask: lower triangular matrix
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size))
            .view(1, 1, config.context_size, config.context_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V in parallel
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = head_size ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Apply attention to values
        y = att @ v

        # Concatenate heads: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.c_proj(y)


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.

    The MLP expands the dimension by 4x, applies GELU, then projects back.
    This is the "position-wise feed-forward network" from the transformer.

    Architecture: Linear(d, 4d) -> GELU -> Linear(4d, d)

    Args:
        config: GPTConfig with model hyperparameters
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-normalization.

    Each block applies:
        1. Layer norm + multi-head self-attention + residual
        2. Layer norm + MLP + residual

    Pre-normalization (norm before attention/MLP) improves training stability
    compared to the original post-normalization design.

    Args:
        config: GPTConfig with model hyperparameters
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block."""
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT-2 language model.

    Combines token embeddings, positional embeddings, transformer blocks,
    and a language modeling head to predict the next token.

    Features:
        - Weight tying between token embedding and output projection
        - Scaled weight initialization following OpenAI's implementation
        - Support for loading pretrained weights from HuggingFace

    Args:
        config: GPTConfig with model hyperparameters

    Example:
        >>> config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
        >>> model = GPT(config)
        >>> tokens = torch.randint(0, 50257, (4, 128))
        >>> logits, _ = model(tokens)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe=nn.Embedding(config.context_size, config.n_embd),  # Position embeddings
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),  # Final layer norm
        ))

        # Language model head (projects to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following GPT-2's scheme.

        - Linear layers: Normal(0, 0.02)
        - Residual projections: Scaled by 1/sqrt(2*n_layer)
        - Embeddings: Normal(0, 0.02)
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                # Scale residual projections for stable training
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            idx: Token indices of shape (batch, seq_len)
            targets: Optional target tokens for computing loss

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.config.context_size, \
            f"Sequence length {T} exceeds context size {self.config.context_size}"

        # Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and projection to vocabulary
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Starting token indices of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens

        Returns:
            Token indices of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context size if needed
            idx_cond = idx if idx.size(1) <= self.config.context_size \
                else idx[:, -self.config.context_size:]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type: str) -> 'GPT':
        """
        Load pretrained GPT-2 weights from HuggingFace.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

        Returns:
            GPT model with pretrained weights
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"Loading pretrained weights: {model_type}")

        # Model configurations
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['context_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter out attention mask buffers
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.mask')]
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')
                      and not k.endswith('.attn.bias')]

        # Weights that need transposition (HF uses Conv1D with transposed weights)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def train(
    model: GPT,
    dataloader,
    n_steps: int,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    log_interval: int = 10
) -> GPT:
    """
    Train a GPT model.

    Args:
        model: GPT model to train
        dataloader: DataLoader or object with next_batch() method
        n_steps: Number of training steps
        optimizer: Optimizer for weight updates
        device: Device to train on ('cpu', 'cuda', 'mps')
        log_interval: Steps between logging

    Returns:
        Trained model
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    for step in range(n_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            print(f'Step {step}: loss={loss.item():.4f}')

    return model
