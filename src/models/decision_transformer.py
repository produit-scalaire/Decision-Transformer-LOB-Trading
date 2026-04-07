import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# optimized for rtx 5090 blackwell
# use with torch compile and bfloat16 autocast

class OptimizedCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # fused projection for max memory bandwidth
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # single matrix multiplication
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # reshape for multi head
        d_k = C // self.n_heads
        q = q.view(B, T, self.n_heads, d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, d_k).transpose(1, 2)
        
        # flash attention dispatch
        # relies on triton kernel under the hood
        # O(N) memory complexity instead of O(N^2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class OptimizedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = OptimizedCausalAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # expansion factor strictly 4 for mlp
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_timestep: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        
        self.embed_timestep = nn.Embedding(max_timestep, d_model)

        # Per-modality LayerNorm equalises embedding scales across RTG (1D),
        # state (state_dim-D), and action modalities — Section 3 of
        # Chen et al. (2021, arXiv:2106.01345).  Without it, the high-dim
        # state projection dominates attention and RTG conditioning is lost.
        self.embed_rtg = nn.Sequential(
            nn.Linear(1, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.embed_action = nn.Sequential(
            nn.Embedding(act_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.predict_action = nn.Linear(d_model, act_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # standard gpt weight initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rtg: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        B, K, _ = states.size()

        # cast implicitly handles bf16 via autocast context in training loop
        time_embeddings = self.embed_timestep(timesteps)
        
        state_emb = self.embed_state(states) + time_embeddings
        action_emb = self.embed_action(actions) + time_embeddings
        rtg_emb = self.embed_rtg(rtg) + time_embeddings

        # vectorized sequence packing
        token_embeddings = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        x = token_embeddings.view(B, 3 * K, self.d_model)

        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)

        # extract state representation to predict next action
        state_representations = x[:, 1::3, :]
        action_logits = self.predict_action(state_representations)
        
        return action_logits