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

class sSplitAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        latent_len = 256,  # context = sequence length - latent
        num_segments,      # number of segments, the context should be divided into
        pos_emb = None,
        dropout = 0.,
        layer_num = 0
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.context_len = self.sequence_len - self.latent_len
        self.num_segments = num_segments
        self.segment_size = self.context_len//num_segments # should divide evenly
        # For example, if context_len = 1536, and num_segments = 2, then
        # segment_size = 768
        assert (self.context_len + latent_len == sequence_len), 'context_length plus latent should be equal to sequence length'
        self.layer_num = layer_num # for PerceiverAR, first layer is different
        self.dim_head = dim//heads  # e.g., 512/8 = 64
        self.scale = self.dim_head ** -0.5  # 1/8

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  # (64)

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  # (512, 512)
        self.to_kv = nn.Linear(dim, dim, bias = False)    # (512, 512)
        self.to_out = nn.Linear(dim, dim) # (512, 512)

       


    def forward(self, x, mask = None): # e.g., x has shape of (4,1024,512)
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 
        # b = 4, n = 1024, *, h = 8, device, causal 
        # x = (4, 1024, 512)
        if self.layer_num ==0:
            x_cxt = x[:,0:self.context_len,:]
            x_lat = x[:,self.context_len:,:]
           

        mask_value = -torch.finfo(x.dtype).max

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  # x = (4, 1024, 512)
        padded_len = x.shape[-2]    # 1024
        # get sequence range, for calculating mask
        seq_range = torch.arange(padded_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)  # b = 4, n = 1024, h = 8, d = 64; 
        # q, kv = [32, 1024, 64]    # (4, 1024, (8x64=512)) --> ((4x8), 1024, 64), h=8
        # rotary embedding
        if exists(self.pos_emb):
            if self.layer_num == 0:
                rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) # (1024, 64)
                rotary_emb = rearrange(rotary_emb, 'n d -> () n d') # (1,1024,64)
                q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q_lat = q[:,self.context_len:,:]
        q_ctx = q[:,0:self.context_len,:]
        q = q_lat
        kv_ctx = kv[:,0:self.context_len,:]
        # scale queries
        q = q * self.scale # (32,1024,64) - each head
        q_ctx = q_ctx * self.scale
        q_ctx_segments = rearrange(q_ctx,'b (s c) d->b s c d',s=self.num_segments)
        lkv = self.norm(kv)
        lkv_ctx = self.norm(kv_ctx)
        lkv_ctx_segments = rearrange(lkv_ctx,'b (s c) d->b s c d',s=self.num_segments)
        lsim = einsum('b i d, b j d -> b i j', q, lkv)  
        #lsim_ctx = einsum('b i d, b j d -> b i j', q_ctx, lkv_ctx)  
        lsim_ctx_segments = einsum('b s i d, b s j d -> b s i j', q_ctx_segments, lkv_ctx_segments) 
        # [32, 256, 1024]; b = 32, (attention of latent_lenxsequence_len e.g., 256x1024)

        # masking
        m_size = lsim.shape[-2]  # 256

        if exists(mask): # called when generate triggers
            mask = rearrange(mask, 'b n -> b () n') # [8, 8, 1, 256]
            lsim.masked_fill_(~mask, mask_value)

        if self.causal:
            causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
            lsim[:,:,self.context_len:].masked_fill_(causal_mask, mask_value)
        
        # softmax for attention
        attn = lsim.softmax(dim = -1)
        attn_ctx = lsim_ctx_segments.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        attn_ctx = self.attn_dropout(attn_ctx)

        # produce output by multiplying with attention, then project out
        out = einsum('b i j, b j d -> b i d', attn, lkv)
        out_ctx_segments = einsum('b s i j, b s j d -> b s i d', attn_ctx, lkv_ctx_segments)
        out_ctx = rearrange(out_ctx_segments,'b s z d-> b (s z) d') # merge context segments
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = h)
        out_ctx = rearrange(out_ctx, '(b h) n d -> b (n) (h d)', h = h)
        out = torch.cat([out_ctx,out],dim=-2)
        return self.to_out(out)


class sSplitAttentionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_tokens,  # for output size determination
        dim,   # embedding
        latent_len,
        num_segments,
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
        self.num_segments = num_segments,
        self.token_emb = nn.Embedding(num_unique_tokens, dim)
        self.dim_head = dim//heads
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, sSplitAttention(dim = dim, heads = heads, sequence_len = sequence_len, latent_len = latent_len, num_segments=num_segments, causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
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
