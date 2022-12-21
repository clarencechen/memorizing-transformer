import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return F.softmax(t, dim = dim)

class KNNAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        num_retrieved_memories = 32,
        xl_max_memories = 0.,
        attn_scale_init = 20,
    ):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))

        self.xl_max_memories = xl_max_memories

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

    def forward(
        self, q, k, v, mask, *,
        knn_memory,
        xl_memory = None,
        add_knn_memory = True,
        rel_pos_bias = None,
        use_causal_mask = False
    ):
        # in paper, they showed normalizing of keys led to more stable training
        # we'll just go with full cosine sim attention https://arxiv.org/abs/2010.04245

        q, k = map(l2norm, (q, k))

        # handle xl memory

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
            k = torch.cat((k_xl_mem, k), dim = -2)
            v = torch.cat((v_xl_mem, v), dim = -2)

        # calculate local attention

        scale = self.scale.exp()

        sim = einsum('b h i d, b j d -> b h i j', q, k) * scale
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        mask_value = -torch.finfo(sim.dtype).max
        sim += mask_value * (1 - mask[:, None, None, :])
        if use_causal_mask:
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # calculate knn attention over memory, if index is passed in

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)

        sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * scale
        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)

        # calculate new XL memories, as well as memories to be discarded

        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        if self.xl_max_memories > 0:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories[:, :-self.xl_max_memories], new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(new_kv_memories_discarded)

        # attention (combining local and distant)

        sim = torch.cat((sim_mem, sim), dim = -1)
        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        local_attn, mem_attn = attn[..., self.num_retrieved_memories:], attn[..., :self.num_retrieved_memories]
        local_out = einsum('b h i j, b j d -> b h i d', local_attn, v)
        mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v)

        out = local_out + mem_out
        return out
