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

class CompressedDoubleAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        latent_len = 256,  # context = sequence length - latent
        compression_size,  # for compressing the history part in first layer
        # second layer onward use the compression_size, as normal history
        pos_emb = None,    # position embedding network i.e., rotary embedding
        dropout = 0.,
        layer_num = 0      # to know if it is the first layer or not
    ):                     
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.compression_size = compression_size  # history to be compressed into
        self.history_len = self.sequence_len - self.latent_len
        assert (self.history_len + latent_len == sequence_len), 'history_length plus latent should be equal to sequence length'
        self.layer_num = layer_num # for handling first layer as it is different
        self.dim_head = dim//heads  
        self.scale = self.dim_head ** -0.5  

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  
        self.to_kv = nn.Linear(dim, dim, bias = False) 
        self.w_out = nn.Linear(dim, dim) 
        self.projection_hist = nn.Linear(self.history_len,self.compression_size)
        # above projection is for compressing the history part in first layer only
       
    def forward(self, x, mask = None): 
        b, n, *_, heads, device, causal = *x.shape, self.heads, x.device, self.causal, 
        mask_value = -torch.finfo(x.dtype).max  # negative infinity to mask out future tokens

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  
        total_len = x.shape[-2]    
        # get sequence range, for calculating mask
        seq_range = torch.arange(total_len, device = device)

        # split q, kv into different heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), qkv)  
        if self.layer_num == 0:
            q_latent = q[:,self.history_len:,:]  
            q_hist = q[:,0:self.history_len,:]
        else:
            q_latent = q[:,self.compression_size:,:]  
            q_hist = q[:,0:self.compression_size,:]
            
        # rotary embedding
        if exists(self.pos_emb): # only first layer has position encoding
            rotary_emb = self.pos_emb(seq_range, cache_key = total_len) 
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
            if self.layer_num == 0:  # kv is full length in first layer
                q_hist = apply_rotary_emb(rotary_emb[:,0:self.history_len,:],q_hist)
                q_latent = apply_rotary_emb(rotary_emb[:,self.history_len:,:],q_latent)
                kv = apply_rotary_emb(rotary_emb,kv)

        # scale queries
        q_latent = q_latent * self.scale # (32,1024,64) - each head
            
        kv = self.norm(kv)
        q_hist = q_hist * self.scale
        if self.layer_num == 0:
            kv_hist = kv[:,0:self.history_len,:]
            att_hist = einsum('b i d, b j d -> b i j', q_hist, kv_hist)
        else:
            kv_hist = kv[:,0:self.compression_size,:]
            att_hist = einsum('b i d, b j d -> b i j', q_hist, kv_hist)

        att = einsum('b i d, b j d -> b i j', q_latent, kv)  # kv is full

        # masking
        m_size = att.shape[-2]  

        if exists(mask): 
            mask = rearrange(mask, 'b n -> b () n') 
            att.masked_fill_(~mask, mask_value)

        if self.causal:
            if self.layer_num == 0:
                causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
                att[:,:,self.history_len:].masked_fill_(causal_mask, mask_value)
            else:
                causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
                att[:,:,self.compression_size:].masked_fill_(causal_mask, mask_value)
 
        
        # final attention
        attn = att.softmax(dim = -1)
        attn_hist = att_hist.softmax(dim=-1)
        
        attn = self.attn_dropout(attn)

        # produce out by multiplying with v
        out = einsum('b i j, b j d -> b i d', attn, kv)
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = heads) # combine heads

        out_hist = einsum('b i j, b j d -> b i d', attn_hist, kv_hist)
        out_hist = rearrange(out_hist, '(b h) n d -> b (n) (h d)', h = heads)
        if self.layer_num == 0: # compress history
            out_hist_rearr = rearrange(out_hist,'b n d->b d n')
            out_hist_proj = self.projection_hist(out_hist_rearr)
            out_hist = rearrange(out_hist_proj,'b d n->b n d')
        out = torch.cat([out_hist,out], dim=1)
        return self.w_out(out)


class CompressedDoubleAttentionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_unique_tokens,  # for output size determination
        dim,   # embedding dimensionality
        latent_len,
        compression_size,
        num_layers,
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
        self.compression_size = compression_size
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CompressedDoubleAttention(dim = dim, heads = heads, sequence_len = sequence_len, latent_len = latent_len,
                  compression_size=compression_size, causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            if attn.fn.layer_num == 0:  
                x = attn(x, mask = mask) + x[:,0:self.compression_size+self.latent_len,:]
            else:
                x = attn(x, mask = mask) + x
            x = ff(x) + x  
        return self.to_logits(x)
