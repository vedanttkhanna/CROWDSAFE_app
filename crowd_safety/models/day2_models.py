"""
Day 2 Models — Social LSTM, Attention LSTM, LSTM Autoencoder, Seq2Seq LSTM

All prediction models share the same interface as Day 1:
    model(obs) -> pred
    obs  : (batch, obs_len,  2)
    pred : (batch, pred_len, 2)

Autoencoder interface:
    model(obs) -> (reconstructed, latent)
    Used for anomaly scoring — NOT trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

OBS_LEN  = 8
PRED_LEN = 12


# ─── 1. Social LSTM ──────────────────────────────────────────────────────────

class SocialLSTM(nn.Module):
    """
    Alahi et al. CVPR 2016.

    Each pedestrian has their own LSTM. At every timestep, each LSTM
    receives a pooled representation of nearby pedestrians' hidden states
    via a grid-based social pooling layer.

    Grid pooling: divide the neighbourhood into an N×N grid.
    Each cell accumulates the hidden states of pedestrians in that cell.
    This gives the model spatial awareness of who is nearby and where.
    """
    def __init__(self, input_dim=2, hidden_dim=64, grid_size=4,
                 neighbourhood_size=2.0, pred_len=PRED_LEN):
        super().__init__()
        self.hidden_dim        = hidden_dim
        self.grid_size         = grid_size
        self.neighbourhood_size = neighbourhood_size
        self.pred_len          = pred_len

        self.input_embed  = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.pool_embed   = nn.Sequential(
            nn.Linear(hidden_dim * grid_size * grid_size, 32), nn.ReLU()
        )

        # Input: position embed (32) + pooled social embed (32)
        self.lstm_cell    = nn.LSTMCell(64, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    def _social_pool(self, hidden_states, positions):
        N  = positions.shape[0]
        H  = self.hidden_dim
        G  = self.grid_size
        ns = self.neighbourhood_size
        pool = torch.zeros(N, G * G * H, device=hidden_states.device)

       # vectorized — no Python loop over pairs
      # rel_pos[i,j] = position[j] - position[i]
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)

       # mask: neighbours within neighbourhood, excluding self
        in_range = (rel_pos.abs().max(dim=-1).values < ns)         # (N, N)
        in_range.fill_diagonal_(False)

        for i in range(N):
            neighbours = in_range[i].nonzero(as_tuple=True)[0]
            if len(neighbours) == 0:
                continue
            for j in neighbours:
                rel = rel_pos[i, j]
                gx  = int(((rel[0].item() + ns) / (2 * ns) * G))
                gy  = int(((rel[1].item() + ns) / (2 * ns) * G))
                gx  = min(max(gx, 0), G - 1)
                gy  = min(max(gy, 0), G - 1)
                cell_idx = gx * G + gy
                pool[i, cell_idx * H:(cell_idx + 1) * H] += hidden_states[j]

        return pool

    def forward(self, obs):
        """
        obs: (B, obs_len, 2)
        Note: Social pooling works best with multiple pedestrians per scene.
        Here we treat the batch as a scene (approximate but trainable).
        """
        B = obs.size(0)
        device = obs.device

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)

        # Encode observed sequence with social pooling at each step
        for t in range(OBS_LEN):
            pos     = obs[:, t, :]                          # (B, 2)
            inp_emb = self.input_embed(pos)                 # (B, 32)
            pool    = self._social_pool(h, pos)             # (B, G*G*H)
            pool_emb = self.pool_embed(pool)                # (B, 32)
            combined = torch.cat([inp_emb, pool_emb], dim=-1)  # (B, 64)
            h, c    = self.lstm_cell(combined, (h, c))

        # Decode future positions
        preds = []
        inp   = obs[:, -1, :]
        for _ in range(self.pred_len):
            inp_emb  = self.input_embed(inp)
            pool     = self._social_pool(h, inp)
            pool_emb = self.pool_embed(pool)
            combined = torch.cat([inp_emb, pool_emb], dim=-1)
            h, c     = self.lstm_cell(combined, (h, c))
            out      = self.output_layer(h)
            preds.append(out)
            inp = out

        return torch.stack(preds, dim=1)                    # (B, 12, 2)


# ─── 2. Attention LSTM ───────────────────────────────────────────────────────

class AttentionLSTM(nn.Module):
    """
    Instead of fixed grid pooling, uses learned attention to weight
    each neighbour's hidden state by relevance.

    Key advantage: attention weights are VISUALIZABLE.
    You can draw lines between pedestrians weighted by attention score —
    showing exactly who is influencing whom in real time.
    This is what makes the Lovable frontend compelling.
    """
    def __init__(self, input_dim=2, hidden_dim=64, pred_len=PRED_LEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        self.input_embed = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())

        # Attention: score how relevant pedestrian j is to pedestrian i
        # Takes concatenation of both hidden states + relative position
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Context from attention + input embed fed to LSTM
        self.lstm_cell    = nn.LSTMCell(32 + hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.last_attn_weights = None   # stored for visualization

    def _attend(self, h_i, h_all, pos_i, pos_all):
        """
        Compute attention-weighted context for pedestrian i.
        h_i    : (H,)    — hidden state of ped i
        h_all  : (N, H)  — hidden states of all peds
        pos_i  : (2,)    — position of ped i
        pos_all: (N, 2)  — positions of all peds
        Returns context : (H,) and weights : (N,)
        """
        N = h_all.size(0)
        h_i_exp  = h_i.unsqueeze(0).expand(N, -1)          # (N, H)
        rel_pos  = pos_all - pos_i.unsqueeze(0)             # (N, 2)
        attn_inp = torch.cat([h_i_exp, h_all, rel_pos], dim=-1)  # (N, 2H+2)
        scores   = self.attn_layer(attn_inp).squeeze(-1)    # (N,)
        weights  = F.softmax(scores, dim=0)                 # (N,)
        context  = (weights.unsqueeze(-1) * h_all).sum(0)   # (H,)
        return context, weights

    def forward(self, obs):
        B      = obs.size(0)
        device = obs.device

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        attn_weights_all = []

        for t in range(OBS_LEN):
            pos     = obs[:, t, :]
            inp_emb = self.input_embed(pos)                 # (B, 32)

            # Compute attention context for each ped
            contexts = torch.zeros(B, self.hidden_dim, device=device)
            step_weights = torch.zeros(B, B, device=device)
            for i in range(B):
                ctx, w = self._attend(h[i], h, pos[i], pos)
                contexts[i] = ctx
                step_weights[i] = w

            attn_weights_all.append(step_weights.detach().cpu())
            combined = torch.cat([inp_emb, contexts], dim=-1)  # (B, 32+H)
            h, c     = self.lstm_cell(combined, (h, c))

        self.last_attn_weights = attn_weights_all           # save for viz

        preds = []
        inp   = obs[:, -1, :]
        for _ in range(self.pred_len):
            inp_emb  = self.input_embed(inp)
            contexts = torch.zeros(B, self.hidden_dim, device=device)
            for i in range(B):
                ctx, _ = self._attend(h[i], h, inp[i], inp)
                contexts[i] = ctx
            combined = torch.cat([inp_emb, contexts], dim=-1)
            h, c     = self.lstm_cell(combined, (h, c))
            out      = self.output_layer(h)
            preds.append(out)
            inp = out

        return torch.stack(preds, dim=1)                    # (B, 12, 2)


# ─── 3. LSTM Autoencoder ─────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Trained ONLY on normal crowd trajectories.
    Encodes an 8-step trajectory to a latent vector, then reconstructs it.

    At inference: high reconstruction error = anomalous movement.
    This is the PRIMARY disruptor flagging mechanism.

    Normal person score  : ~0.05 - 0.15
    Anomalous person     : ~0.40 - 0.80+

    Anomaly types it catches:
      - Sudden stops in dense flow
      - Counter-flow walking
      - Erratic direction changes
      - Abnormal speed (too fast or too slow)
    """
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_embed = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())

        # Encoder
        self.encoder_lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder — reconstructs the INPUT sequence (not future)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm     = nn.LSTM(32, hidden_dim, batch_first=True)
        self.output_layer     = nn.Linear(hidden_dim, input_dim)

    def encode(self, obs):
        """obs: (B, 8, 2) → latent: (B, latent_dim)"""
        emb = self.input_embed(obs)
        _, (h, _) = self.encoder_lstm(emb)
        return self.hidden_to_latent(h.squeeze(0))

    def decode(self, latent, seq_len):
        """latent: (B, latent_dim) → reconstructed: (B, seq_len, 2)"""
        B      = latent.size(0)
        device = latent.device
        h      = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        c      = torch.zeros_like(h)

        # Feed zeros as decoder input (unconditional reconstruction)
        dec_inp = torch.zeros(B, seq_len, 32, device=device)
        out, _  = self.decoder_lstm(dec_inp, (h, c))
        return self.output_layer(out)                       # (B, seq_len, 2)

    def forward(self, obs):
        latent = self.encode(obs)
        recon  = self.decode(latent, obs.size(1))
        return recon, latent

    def anomaly_score(self, obs):
        """
        Returns per-sample reconstruction error.
        Shape: (B,) — one score per pedestrian trajectory.
        Higher = more anomalous.
        """
        recon, _ = self.forward(obs)
        error = torch.norm(recon - obs, dim=-1).mean(dim=-1)  # (B,)
        return error


# ─── 4. Seq2Seq LSTM ─────────────────────────────────────────────────────────

class Seq2SeqLSTM(nn.Module):
    """
    Encoder reads observed trajectory → context vector.
    Decoder generates predicted future.

    Divergence scoring: at inference time, compare the model's
    predicted trajectory against what actually happened.
    Large divergence = person behaved unexpectedly = risk signal.

    This is FORWARD-LOOKING risk — you catch the disruption
    as it starts, not after it's expressed.

    Also used for trajectory prediction (same interface as other models).
    """
    def __init__(self, input_dim=2, hidden_dim=64, pred_len=PRED_LEN,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_embed  = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.encoder      = nn.LSTM(32, hidden_dim, num_layers=num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.decoder      = nn.LSTM(32, hidden_dim, num_layers=num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, obs, target=None, teacher_forcing=0.0):
        """
        obs     : (B, obs_len, 2)
        target  : (B, pred_len, 2) optional — for teacher forcing during training
        Returns : (B, pred_len, 2)
        """
        B      = obs.size(0)
        device = obs.device

        # Encode
        obs_emb = self.input_embed(obs)
        _, (h, c) = self.encoder(obs_emb)

        # Decode
        preds  = []
        inp    = obs[:, -1, :]
        for t in range(self.pred_len):
            inp_emb          = self.input_embed(inp).unsqueeze(1)    # (B,1,32)
            out, (h, c)      = self.decoder(inp_emb, (h, c))
            pred             = self.output_layer(out.squeeze(1))     # (B, 2)
            preds.append(pred)

            # Teacher forcing: sometimes feed ground truth as next input
            if target is not None and torch.rand(1).item() < teacher_forcing:
                inp = target[:, t, :]
            else:
                inp = pred

        return torch.stack(preds, dim=1)                             # (B,12,2)

    def divergence_score(self, obs, actual_future):
        """
        Compute how much the actual future diverged from prediction.
        obs           : (B, 8,  2)
        actual_future : (B, 12, 2)
        Returns       : (B,) divergence score per pedestrian
        """
        with torch.no_grad():
            predicted = self.forward(obs)
        diff = torch.norm(predicted - actual_future, dim=-1)  # (B, 12)
        return diff.mean(dim=-1)                               # (B,)


# ─── Model registry ──────────────────────────────────────────────────────────

def get_model(name, **kwargs):
    models = {
        "social":      SocialLSTM,
        "attention":   AttentionLSTM,
        "autoencoder": LSTMAutoencoder,
        "seq2seq":     Seq2SeqLSTM,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models.keys())}")
    return models[name](**kwargs)