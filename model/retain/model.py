
import os

import torch
import torch.nn as  nn

from typing import Optional,  Tuple, List
import torch.nn.utils.rnn as rnn_utils



class RETAINLayer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)

        self.alpha_li = nn.Linear(feature_size, 1)
        self.beta_li = nn.Linear(feature_size, feature_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )

        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()

        rx = self.reverse_x(x, lengths)

        try:
            attn_alpha = self.compute_alpha(rx, lengths)
        except Exception as e:
            return  torch.sum(x, dim=1)
        #attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)

        x = x[:,:attn_alpha.shape[1],:]
        c = attn_alpha * attn_beta * x  # (patient, sequence len, feature_size)
        c = torch.sum(c, dim=1)  # (patient, feature_size)
        return c

class RETAIN(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
                 embedding_dim: int = 16, dropout=0.5, e_best=None):
        super(RETAIN, self).__init__()
        self.embedding_dim = embedding_dim
        Tokenizers = {k: Tokenizers[k] for k in list(Tokenizers)[0:]}
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()

        self.feature_keys = ['cond_hist']

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout=dropout)

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.device = device

        self.visit_emb_save_path = './visit_embeddings'
        os.makedirs(self.visit_emb_save_path, exist_ok=True)
        self.ccs_emb_dict = {}

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def forward(self, batchdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)

            x_emb = self.embeddings[feature_key](x)

            x = torch.sum(x_emb, dim=2)

            mask = torch.sum(x, dim=2) != 0
            x = self.retain[feature_key](x, mask)

            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)

        logits = self.fc(patient_emb)
        return logits
