
## 简单看看 

【真-极简爬坡式强化学习入门(代码现编，PyTorch版）】 https://www.bilibili.com/video/BV1Gq4y1v7Bs/?p=2&share_source=copy_web&vd_source=ec8629ab079093aa739549702649bab1




# Causal Attention: Mathematical Formulation & Implementation

## Mathematical Formulation

### Standard Self-Attention
The standard self-attention mechanism computes:

$$
Attention(Q, K, V) = softmax\left( \frac{QK^T}{\sqrt{d_{k}}} \right)V
$$



Where:
- **Q** (Query): d_model × d_k matrix
- **K** (Key): d_model × d_k matrix  
- **V** (Value): d_model × d_v matrix
- **d_k**: dimension of key/query vectors
- **$\sqrt{d_{k}  }$**: scaling factor to prevent softmax saturation

### Causal (Masked) Self-Attention
For causal attention, we apply a mask before the softmax:

```
Attention(Q, K, V) = softmax(mask(QK^T / √d_k))V
```

The mask function applies:
```
mask(S_ij) = {
    S_ij    if j ≤ i  (past/current positions)
    -∞      if j > i  (future positions)
}
```

### Step-by-Step Mathematical Process

1. **Compute attention scores**: `S = QK^T / √d_k`
2. **Apply causal mask**: `S_masked = mask(S)`
3. **Apply softmax**: `A = softmax(S_masked)`
4. **Compute output**: `O = AV`

The causal mask ensures that `softmax(-∞) = 0`, effectively removing future positions from attention.

## Implementation Examples

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Create causal mask (lower triangular matrix)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, n_heads, seq_len, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: (batch, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # Shape: (batch, n_heads, seq_len, d_k)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        return self.W_o(out)

# Example usage
model = CausalSelfAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch_size, seq_len, d_model)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### NumPy Implementation (Educational)

```python
import numpy as np

def causal_self_attention(x, W_q, W_k, W_v, W_o):
    """
    Simplified single-head causal self-attention
    
    Args:
        x: input sequence (seq_len, d_model)
        W_q, W_k, W_v: weight matrices (d_model, d_model)
        W_o: output projection (d_model, d_model)
    """
    seq_len, d_model = x.shape
    d_k = d_model  # For simplicity, assume d_k = d_model
    
    # Step 1: Compute Q, K, V
    Q = np.dot(x, W_q)  # (seq_len, d_model)
    K = np.dot(x, W_k)
    V = np.dot(x, W_v)
    
    # Step 2: Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Step 3: Create and apply causal mask
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(causal_mask == 1, scores, -np.inf)
    
    # Step 4: Apply softmax
    attn_weights = softmax(scores, axis=-1)
    
    # Step 5: Apply attention to values
    out = np.dot(attn_weights, V)  # (seq_len, d_model)
    
    # Step 6: Output projection
    return np.dot(out, W_o)

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Example usage
seq_len, d_model = 5, 4
x = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model) 
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

output = causal_self_attention(x, W_q, W_k, W_v, W_o)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## Key Implementation Details

### 1. Causal Mask Creation
```python
# Lower triangular matrix (1s below and on diagonal, 0s above)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

# Example for seq_len=4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0], 
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

### 2. Mask Application
```python
# Set future positions to -infinity before softmax
scores = scores.masked_fill(mask == 0, float('-inf'))
# After softmax: exp(-inf) = 0, so future positions get 0 attention
```

### 3. Multi-Head Attention
The same causal mask is applied to all attention heads simultaneously.

## Computational Complexity

- **Time Complexity**: O (n²d) where n is sequence length, d is model dimension
- **Space Complexity**: O (n²) for storing attention weights
- **Mask Storage**: O (n²) but can be precomputed and reused

## Applications

1. **Language Models**: GPT, GPT-2, GPT-3/4 use causal attention
2. **Text Generation**: Ensures autoregressive property
3. **Decoder Blocks**: In encoder-decoder architectures (Transformers)
4. **Time Series**: When future information shouldn't influence past predictions