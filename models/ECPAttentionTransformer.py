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

class ECPAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        num_segments,      # number of segments, the seq should be divided into
        pos_emb = None,
        dropout = 0.,
        layer_num = 0
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.segment_len = sequence_len//num_segments
        self.num_segments = num_segments
        # For example, if context_len = 1024, and num_segments = 8, then
        # segment_len = 128
        assert (self.segment_len * num_segments == sequence_len), 'segment_length times num_segments should be equal to sequence length'
        self.layer_num = layer_num 
        self.dim_head = dim//heads  
        self.scale = self.dim_head ** -0.5  
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        self.norm = nn.LayerNorm(self.dim_head)  
        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, dim, bias = False)  
        self.to_kv = nn.Linear(dim, dim, bias = False)    
        self.to_out = nn.Linear(dim, dim) 
        self.to_out_0 = nn.Linear(dim, dim) 

    def forward(self, x, mask = None): # e.g., x has shape of (4,1024,512)
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 
        # b = batch, n = seuence_len, h = number of heads 

        mask_value = -torch.finfo(x.dtype).max

        # queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  
        seq_len = x.shape[-2]   
        # get sequence range, for calculating position encoding
        seq_range = torch.arange(seq_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)  
        # rotary embedding
        if exists(self.pos_emb):
            if self.layer_num == 0:
                rotary_emb = self.pos_emb(seq_range, cache_key = seq_len) 
                rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
                q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q = q * self.scale # scale queries
        qs = rearrange(q, 'c (s m) d ->c s m d', s = self.num_segments) 
        kvs = rearrange(kv,'c (s m) d ->c s m d', s = self.num_segments)
        # catenate consecutive statements for kv
        fcat = lambda t : torch.cat([kvs[:,t,:,:],kvs[:,t+1,:,:]], dim=1).unsqueeze(dim=1)
        kvs_ccs = torch.cat([fcat(t) for t in range(0,self.num_segments-1)],dim=1)
        qs_ccs = qs[:,1:self.num_segments,:,:]
        lkv = self.norm(kvs_ccs)
        lsim = einsum('c s m d, c s n d -> c s m n', qs_ccs, kvs_ccs) # attention in segments
        # masking
        m_size = lsim.shape[-2]  

        if exists(mask): 
            mask = rearrange(mask, 'b n -> b () n') 
            lsim.masked_fill_(~mask, mask_value)

        if self.causal:
            causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
            lsim[:,:,:,self.segment_len:].masked_fill_(causal_mask, mask_value)
        
        # attention and masking in very first segment
        attn0 = einsum('c s m d, c s n d->c s m n',qs[:,0:1,:,:],kvs[:,0:1,:,:])
        attn0.masked_fill_(causal_mask, mask_value)
  
        
        # final attention
        attn_0= attn0.softmax(dim=-1)   # attention in first segment
        attn_1 = lsim.softmax(dim = -1) # attention in remaining segments
        attnd_0 = self.attn_dropout(attn_0)
        attnd_1 = self.attn_dropout(attn_1)

        out0 = einsum('c s i j, c s j d -> c s i d', attnd_0, kvs[:,0:1,:,:])
        out1 = einsum('c s i j, c s j d -> c s i d', attnd_1, kvs_ccs)
 
        out1 = rearrange(out1,'c s z d-> c (s z) d') # merge remaining segments  
        out_1 = rearrange(out1, '(b h) n d -> b (n) (h d)', h = h)  # combine heads
        out_1o = self.to_out(out_1) # projection of combined heads for remaining segments
        out0 = rearrange(out0,'c s z d-> c (s z) d') # merge segments
        out_0 = rearrange(out0, '(b h) n d -> b (n) (h d)', h = h)
        out_0o = self.to_out_0(out_0) # projection of first segment
        out = torch.cat([out_0o, out_1o], dim = 1)
        return out


class ECPAttentionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_tokens,  # for output size determination
        dim,   # embedding dimesionality
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
        self.segment_len = sequence_len//num_segments
        self.num_segments = num_segments
        self.token_emb = nn.Embedding(num_unique_tokens, dim)
        self.dim_head = dim//heads
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ECPAttention(dim = dim, heads = heads, sequence_len = sequence_len, num_segments=num_segments, causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
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
