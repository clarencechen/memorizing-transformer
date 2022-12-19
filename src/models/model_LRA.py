
import torch
import torch.nn as nn
import math

from models.model_transformer import Model

def pooling(inp, mode):
    if mode == "CLS":
        pooled = inp[:, 0, :]
    elif mode == "MEAN":
        pooled = inp.mean(dim = 1)
    else:
        raise Exception()
    return pooled

def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask

class SCHead(nn.Module):
    def __init__(self, config, dual_input=False):
        super().__init__()
        self.dual_input = dual_input
        self.pooling_mode = config["pooling_mode"]
        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"] * (4 if dual_input else 1), config["transformer_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
        )

    def forward(self, inp_0, inp_1=None):
        X_0 = pooling(inp_0, self.pooling_mode)
        if self.dual_input and inp_1 is not None:
            X_1 = pooling(inp_1, self.pooling_mode)
            seq_score = self.mlpblock(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim = -1))
        else:
            seq_score = self.mlpblock(X_0)
        return seq_score

class ModelForSC(nn.Module):
    def __init__(self, config, batch_size=None, dual_input=False):
        super().__init__()

        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]
        self.dual_input = dual_input

        self.model = Model(config)

        self.seq_classifer = SCHeadDual(config) if dual_input else SCHead(config)
        if config["attn_type"].startswith("memorizing"):
            assert isinstance(batch_size, int), "Batch size must be provided to instantiate KNN Memories for Memorizing Transformer"
            self.knn_memories = KNNMemoryList.create_memories(
            batch_size=batch_size, num_memory_layers=config["num_layers"],
            )(
                dim=config["head_dim"], max_memories=config["max_seq_len"],
                multiprocessing=config["knn_memory_multiprocessing"],
            )

    def forward(self, *, input_ids_0, mask_0, label, input_ids_1=None, mask_1=None):

        if self.pooling_mode == "CLS":
            input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
            if self.dual_input and input_ids_1 is not None and mask_1 is not None:
                input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

        if config["attn_type"].startswith("memorizing"):
            for input_ids, mask in ((input_ids_0, mask_0), (input_ids_1, mask_1)):
                token_out = []
                if input_ids is None and mask is None:
                    token_out.append(None)
                    continue
                token_segments_out = []
                num_segments = math.ceil(config["max_seq_len"] / config["segment_len"])
                for seq, mask in zip(input_ids_0.chunk(num_segments, dim=-1), mask_0.chunk(num_segments, dim=-1)):
                    token_segments_out.append(self.model(seq, mask, knn_memories=self.knn_memories))
                token_out.append(torch.cat(token_segments_out, dim=-2))
                self.knn_memories.clear_memory()
            token_out_0, token_out_1 = token_out
        else:
            token_out_0 = self.model(input_ids_0, mask_0)
            if self.dual_input and input_ids_1 is not None and mask_1 is not None:
                token_out_1 = self.model(input_ids_1, mask_1)

        if self.dual_input and token_out_1 is not None:
            seq_scores = self.seq_classifer(token_out_0, token_out_1)
        else:
            seq_scores = self.seq_classifer(token_out_0)
        seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
        seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
        outputs = {}
        outputs["loss"] = seq_loss
        outputs["accu"] = seq_accu

        return outputs
