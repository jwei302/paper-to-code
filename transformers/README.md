# GPT-2 Transformer Implementation

A PyTorch implementation of the GPT-2 architecture from the paper ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019).

## Paper Overview

GPT-2 is a **decoder-only transformer** trained on a large corpus of web text. It demonstrates that language models can learn to perform various NLP tasks without explicit supervision, purely through next-token prediction.

### Key Contributions
- **Scale**: Showed that larger models with more data continue to improve
- **Zero-shot learning**: Can perform tasks like translation and summarization without task-specific training
- **Unsupervised multitask learning**: Single model learns multiple capabilities

### Architecture

```
Input Tokens
    │
    ▼
[Token Embedding] + [Position Embedding]
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Block (x12)            │
│  ┌─────────────────────────────────┐│
│  │ LayerNorm                       ││
│  │     │                           ││
│  │ Multi-Head Causal Attention     ││
│  │     │                           ││
│  │ + ←─┘ (residual)                ││
│  │     │                           ││
│  │ LayerNorm                       ││
│  │     │                           ││
│  │ MLP (4x expansion, GELU)        ││
│  │     │                           ││
│  │ + ←─┘ (residual)                ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    │
    ▼
[Layer Norm]
    │
    ▼
[Linear → vocab_size]
    │
    ▼
Output Logits
```

Key architectural details:
- **Pre-normalization**: LayerNorm applied before attention/MLP (not after in Attention paper). 
- **GELU activation**: Smoother than ReLU, used in feed-forward layers. 
- **Weight tying**: Token embedding shared with output projection, saving memory. 
- **Causal masking**: Prevents attending to future tokens, allowing autoregressive generation

## Usage

### Minimum Working Example

```python
import torch
from transformer import GPT, GPTConfig, train

# Create a small GPT model
config = GPTConfig(
    context_size=256,
    vocab_size=50257,
    n_layer=6,
    n_head=6,
    n_embd=384
)
model = GPT(config)

# Forward pass
tokens = torch.randint(0, 50257, (4, 128))  # Batch of 4, seq len 128
logits, loss = model(tokens)  # logits: (4, 128, 50257)

# With targets (for training)
targets = torch.randint(0, 50257, (4, 128))
logits, loss = model(tokens, targets)
print(f"Loss: {loss.item()}")

# Text generation
prompt = torch.randint(0, 50257, (1, 10))  # Starting tokens
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
```

### Loading Pretrained Weights

```python
from transformer import GPT

# Load pretrained GPT-2 from HuggingFace
model = GPT.from_pretrained('gpt2')  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# Generate text
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = torch.tensor(enc.encode("Hello, world")).unsqueeze(0)
output = model.generate(tokens, max_new_tokens=50, temperature=0.8, top_k=40)
print(enc.decode(output[0].tolist()))
```

### Training Loop

```python
from transformer import GPT, GPTConfig, train

config = GPTConfig()
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model = train(
    model=model,
    dataloader=train_loader,  # Must have next_batch() method
    n_steps=1000,
    optimizer=optimizer,
    device='cuda',
    log_interval=50
)
```

## Model Sizes

| Model | Layers | Heads | d_model | Parameters |
|-------|--------|-------|---------|------------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1.5B |

## Files

- `transformer.py` - Model architecture, training, and generation functions

## Requirements

- PyTorch >= 1.0
- tiktoken (for tokenization)
- transformers (optional, for loading pretrained weights)

## References

```bibtex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  year={2019}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={NeurIPS},
  year={2017}
}
```
