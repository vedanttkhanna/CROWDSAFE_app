
"""
Day 1 Models — Vanilla LSTM, Bidirectional LSTM, Stacked LSTM
All share the same interface: model(obs) -> pred
obs  : (batch, obs_len,  2)
pred : (batch, pred_len, 2)
"""

import torch
import torch.nn as nn

OBS_LEN  = 8
PRED_LEN = 12


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.encoder     = nn.LSTM(32, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(32, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, obs):
        obs_emb = self.input_embed(obs)
        _, (h, c) = self.encoder(obs_emb)
        h = h.squeeze(0)
        c = c.squeeze(0)
        preds = []
        inp = obs[:, -1, :]
        for _ in range(self.pred_len):
            inp_emb = self.input_embed(inp)
            h, c    = self.decoder_cell(inp_emb, (h, c))
            out     = self.output_layer(h)
            preds.append(out)
            inp = out
        return torch.stack(preds, dim=1)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim
        self.input_embed  = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.encoder      = nn.LSTM(32, hidden_dim, batch_first=True, bidirectional=True)
        self.proj         = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_cell = nn.LSTMCell(32, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, obs):
        obs_emb = self.input_embed(obs)
        _, (h, c) = self.encoder(obs_emb)
        h = torch.tanh(self.proj(torch.cat([h[0], h[1]], dim=-1)))
        c = torch.tanh(self.proj(torch.cat([c[0], c[1]], dim=-1)))
        preds = []
        inp = obs[:, -1, :]
        for _ in range(self.pred_len):
            inp_emb = self.input_embed(inp)
            h, c    = self.decoder_cell(inp_emb, (h, c))
            out     = self.output_layer(h)
            preds.append(out)
            inp = out
        return torch.stack(preds, dim=1)


class StackedLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim
        self.input_embed  = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.encoder      = nn.LSTM(32, hidden_dim, num_layers=num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.decoder_cell = nn.LSTMCell(32, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, obs):
        obs_emb = self.input_embed(obs)
        _, (h, c) = self.encoder(obs_emb)
        h = h[-1]
        c = c[-1]
        preds = []
        inp = obs[:, -1, :]
        for _ in range(self.pred_len):
            inp_emb = self.input_embed(inp)
            h, c    = self.decoder_cell(inp_emb, (h, c))
            out     = self.output_layer(h)
            preds.append(out)
            inp = out
        return torch.stack(preds, dim=1)


def get_model(name, **kwargs):
    models = {
        "vanilla": VanillaLSTM,
        "bilstm":  BidirectionalLSTM,
        "stacked": StackedLSTM,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models.keys())}")
    return models[name](**kwargs)
