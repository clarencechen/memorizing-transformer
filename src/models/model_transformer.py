
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint

from models.attention import Attention

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        
        
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask, knn_memories=None, add_knn_memory=True):
        attn_kwargs = {}
        if knn_memories is not None:
            attn_kwargs = {'knn_memory': knn_memories, 'add_knn_memory': add_knn_memory}
        X = self.dropout1(self.mha(self.norm1(X), mask, **attn_kwargs)) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

def compute_batch_indices_to_clear(x, token_id):
    """ clears the KNN memories based on if the batch row contains the specified token id """
    """ for auto-clearing KNN memories based on start and end of strings """

    clear_memory = (x == token_id).any(dim = -1)
    batch_indices, _ = clear_memory.nonzero(as_tuple = True)
    batch_indices_to_clear = batch_indices.tolist()

    if len(batch_indices_to_clear) == 0:
        return
    return batch_indices_to_clear

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = TransformerLayer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", TransformerLayer(config))

        self.norm = nn.LayerNorm(config["transformer_dim"])

        if config["attn_type"].startswith("memorizing"):            
            # memory layer shifting from a little known paper https://arxiv.org/abs/2012.15688
            self.shift_knn_memories_down = config["shift_knn_memories_down"]

    def forward(self, input_ids, mask = None, knn_memories = None, add_knn_memory = True):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)

        knn_memories_iter = None
        if knn_memories is not None:
            batch_size = input_ids.shape[0]
            # validate KNN memories to have enough indices for batch size
            assert all([memory.num_indices == batch_size for memory in knn_memories]), f'you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices'
            # shifting memories a number of layers down, little known technique shown to enhance memories from Ernie-Doc paper
            if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
                knn_memories = [*knn_memories[self.shift_knn_memories_down:], *knn_memories[:self.shift_knn_memories_down]]
            # iterate through the memories in order of the ascending layers that contain KNNAttention
            knn_memories_iter = iter(knn_memories)

        if self.tied_weights:
            for idx in range(self.num_layers):
                layer_memories = None if knn_memories_iter is None else next(knn_memories_iter)
                X = self.transformer(X, mask, knn_memories=layer_memories, add_knn_memory=add_knn_memory)
        else:
            for idx in range(self.num_layers):
                layer_memories = None if knn_memories_iter is None else next(knn_memories_iter)
                X = getattr(self, f"transformer_{idx}")(X, mask, knn_memories=layer_memories, add_knn_memory=add_knn_memory)


        X = self.norm(X) * mask[:, :, None]

        return X
