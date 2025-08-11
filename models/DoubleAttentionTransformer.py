from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from models.Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat
import numpy as np


# -----------helper functions---------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class DoubleAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        latent_len = 256,  # context = sequence length - latent
        pos_emb = None,
        dropout = 0.05,
        layer_num = 0
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.context_len = self.sequence_len - self.latent_len
        assert (self.context_len + latent_len == sequence_len), 'context_length plus latent should be equal to sequence length'
        self.layer_num = layer_num 
        self.dim_head = dim//heads  
        self.scale = self.dim_head ** -0.5  

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  
        self.to_kv = nn.Linear(dim, dim, bias = False)    
        self.to_out = nn.Linear(dim, dim) 

       


    def forward(self, x, mask = None): 
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 
        if self.layer_num ==0:
            x_cxt = x[:,0:self.context_len,:]
            x_lat = x[:,self.context_len:,:]

        mask_value = -torch.finfo(x.dtype).max

        # queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  # x = (4, 1024, 512)
        seq_len = x.shape[-2]    
        # get sequence range, for calculating position encoding
        seq_range = torch.arange(seq_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)  
        # rotary embedding
        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = seq_len) 
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
            q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q_lat = q[:,self.context_len:,:]
        q_ctx = q[:,0:self.context_len,:]
        q = q_lat
        kv_ctx = kv[:,0:self.context_len,:]
        # scale queries
        q = q * self.scale 
        q_ctx = q_ctx * self.scale
        lkv = self.norm(kv)
        lkv_ctx = self.norm(kv_ctx)
        lsim = einsum('b i d, b j d -> b i j', q, lkv)  
        lsim_ctx = einsum('b i d, b j d -> b i j', q_ctx, lkv_ctx)  
 
        # masking
        m_size = lsim.shape[-2]  

        if exists(mask): 
            mask = rearrange(mask, 'b n -> b () n') 
            lsim.masked_fill_(~mask, mask_value)

        if self.causal:
            causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
            lsim[:,:,self.context_len:].masked_fill_(causal_mask, mask_value)
        
        # final attention
        attn = lsim.softmax(dim = -1)
        attn_ctx = lsim_ctx.softmax(dim=-1)  # context or history attention
        attn = self.attn_dropout(attn)  # for latent part
        attn_ctx = self.attn_dropout(attn_ctx) # for context part

        # compute output by multiplying attention with v and project out
        out = einsum('b i j, b j d -> b i d', attn, lkv)
        out_ctx = einsum('b i j, b j d -> b i d', attn_ctx, lkv_ctx)
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = h)
        out_ctx = rearrange(out_ctx, '(b h) n d -> b (n) (h d)', h = h)
        out = torch.cat([out_ctx,out],dim=-2)  # combine context and latent parts
        return self.to_out(out)


class DoubleAttentionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_tokens,  # for output size determination
        dim,   # embedding
        latent_len,
        num_layers, # number of layers
        heads = 8,
        sequence_len,
        causal = True,
        ff_mult = 4,  # expand feedforward by 4 times then reduce back to embedding size
        ff_dropout = 0.05,
        attn_dropout = 0.05
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.token_emb = nn.Embedding(num_unique_tokens, dim)
        self.dim_head = dim//heads
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DoubleAttention(dim = dim, heads = heads, sequence_len = sequence_len, latent_len = latent_len,causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x   
        return self.to_logits(x)
