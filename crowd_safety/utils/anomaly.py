"""
Anomaly & Risk Scoring Engine

Takes outputs from all 6 models and computes:
  1. Per-individual anomaly scores  (who is the disruptor)
  2. Zone-level crowd risk scores   (where is danger forming)
  3. Stampede precursor signals     (is the crowd about to become dangerous)

Risk score is 0-100:
  0  - 30  : Normal flow
  30 - 60  : Monitor — elevated density or velocity conflict
  60 - 80  : Warning — multiple signals elevated
  80 - 100 : Critical — stampede precursor detected
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class PedestrianState:
    ped_id:          int
    positions:       np.ndarray      # (obs_len, 2) observed positions
    predicted:       np.ndarray      # (pred_len, 2) predicted future
    velocity:        np.ndarray      # (2,) current velocity vector
    speed:           float
    anomaly_score:   float = 0.0     # from autoencoder
    divergence_score: float = 0.0    # from seq2seq
    is_flagged:      bool  = False
    flag_reasons:    List[str] = field(default_factory=list)


@dataclass
class CrowdRiskReport:
    overall_risk:      float          # 0-100
    density_risk:      float          # 0-100
    velocity_risk:     float          # 0-100
    anomaly_risk:      float          # 0-100
    flagged_ids:       List[int]      = field(default_factory=list)
    risk_label:        str            = "Normal"
    zone_densities:    Dict           = field(default_factory=dict)
    precursor_signals: List[str]      = field(default_factory=list)


# ─── Individual anomaly scoring ───────────────────────────────────────────────

def score_reconstruction(autoencoder, obs_tensor, threshold=0.25):
    """
    Run LSTM Autoencoder on each pedestrian's observed trajectory.
    Returns anomaly scores (B,) and flags (B,).

    threshold: reconstruction error above this = flagged
    Typical values:
      Normal flow     : 0.05 - 0.15
      Unusual movement: 0.25 - 0.50
      Severe anomaly  : 0.50+
    """
    autoencoder.eval()
    with torch.no_grad():
        scores = autoencoder.anomaly_score(obs_tensor)   # (B,)
    flags = scores > threshold
    return scores.cpu().numpy(), flags.cpu().numpy()


def score_divergence(seq2seq_model, obs_tensor, future_tensor, threshold=0.50):
    """
    Compare seq2seq prediction vs actual future trajectory.
    High divergence = person behaved unexpectedly.
    Returns divergence scores (B,) and flags (B,).
    """
    seq2seq_model.eval()
    scores = seq2seq_model.divergence_score(obs_tensor, future_tensor)
    flags  = scores > threshold
    return scores.cpu().numpy(), flags.cpu().numpy()


# ─── Crowd-level metrics ──────────────────────────────────────────────────────

def compute_local_density(positions, radius=2.0):
    """
    For each pedestrian, count how many others are within `radius` metres.
    positions: (N, 2) numpy array of current positions
    Returns  : (N,) density count per pedestrian

    Danger threshold (Helbing): > 6 people/m²
    For radius=2m, circle area = 12.56m², so danger count ≈ 75 neighbours
    Practical warning threshold: > 8 neighbours within 2m
    """
    N       = positions.shape[0]
    density = np.zeros(N)
    for i in range(N):
        dists      = np.linalg.norm(positions - positions[i], axis=1)
        density[i] = np.sum(dists < radius) - 1   # exclude self
    return density


def compute_velocity_consensus(positions, velocities, radius=2.0):
    """
    For each pedestrian, measure velocity variance among nearby neighbours.
    High variance = conflicting flows = crowd turbulence precursor.

    positions : (N, 2)
    velocities: (N, 2)
    Returns   : (N,) velocity conflict score per pedestrian
    """
    N        = positions.shape[0]
    conflict = np.zeros(N)
    for i in range(N):
        dists    = np.linalg.norm(positions - positions[i], axis=1)
        mask     = (dists < radius) & (dists > 0)
        if mask.sum() < 2:
            continue
        neighbour_vels = velocities[mask]
        # Variance in direction (angle) among neighbours
        angles   = np.arctan2(neighbour_vels[:, 1], neighbour_vels[:, 0])
        conflict[i] = np.std(angles)   # high std = conflicting directions
    return conflict


def compute_counter_flow_ratio(velocities, zone_size=5.0):
    """
    Detect bidirectional flow conflict — a known stampede precursor.
    Returns fraction of pedestrians moving against dominant flow.

    velocities: (N, 2)
    Returns   : float 0-1, fraction moving counter to dominant direction
    """
    if len(velocities) < 2:
        return 0.0
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    dominant = np.median(angles)
    # Counter-flow: angle difference > 90 degrees from dominant
    diffs = np.abs(angles - dominant)
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    return float(np.mean(diffs > np.pi / 2))


def compute_speed_anomaly(velocities, baseline_mean=1.4, baseline_std=0.4):
    """
    Flag pedestrians moving significantly faster or slower than normal.
    baseline_mean: normal walking speed in m/s (~1.4 m/s)
    baseline_std : standard deviation of normal walking speeds
    Returns      : (N,) z-score of each pedestrian's speed
    """
    speeds  = np.linalg.norm(velocities, axis=1)              # (N,)
    z_scores = np.abs(speeds - baseline_mean) / baseline_std
    return z_scores


# ─── Master risk scorer ───────────────────────────────────────────────────────

def compute_risk_score(
    pedestrian_states: List[PedestrianState],
    autoencoder       = None,
    seq2seq_model     = None,
    obs_tensor        = None,
    future_tensor     = None,
) -> CrowdRiskReport:
    """
    Combine all signals into a single crowd risk report.

    Signal weights (tunable):
      density_weight   = 0.35   — most reliable predictor
      velocity_weight  = 0.25   — direction conflict
      anomaly_weight   = 0.25   — individual anomalies
      counterflow_weight = 0.15 — bidirectional flow
    """
    if not pedestrian_states:
        return CrowdRiskReport(overall_risk=0, density_risk=0,
                               velocity_risk=0, anomaly_risk=0)

    N          = len(pedestrian_states)
    positions  = np.array([p.positions[-1] for p in pedestrian_states])
    velocities = np.array([p.velocity for p in pedestrian_states])

    # ── 1. Density risk ──
    densities    = compute_local_density(positions, radius=2.0)
    max_density  = densities.max() if len(densities) > 0 else 0
    # Normalize: 0 neighbours = 0 risk, 20+ neighbours = 100 risk
    density_risk = float(min(max_density / 20.0 * 100, 100))

    # ── 2. Velocity consensus risk ──
    conflicts     = compute_velocity_consensus(positions, velocities)
    # Max conflict score of ~π = 100 risk
    velocity_risk = float(min(conflicts.max() / np.pi * 100, 100)) if len(conflicts) > 0 else 0

    # ── 3. Counter-flow risk ──
    cf_ratio      = compute_counter_flow_ratio(velocities)
    counterflow_risk = cf_ratio * 100

    # ── 4. Individual anomaly risk ──
    anomaly_scores = np.array([p.anomaly_score for p in pedestrian_states])
    anomaly_risk   = float(min(anomaly_scores.mean() * 200, 100))

    # ── 5. Combined risk score ──
    overall_risk = (
        0.35 * density_risk +
        0.25 * velocity_risk +
        0.25 * anomaly_risk +
        0.15 * counterflow_risk
    )
    overall_risk = float(min(overall_risk, 100))

    # ── 6. Risk label ──
    if overall_risk < 30:
        label = "Normal"
    elif overall_risk < 60:
        label = "Monitor"
    elif overall_risk < 80:
        label = "Warning"
    else:
        label = "CRITICAL"

    # ── 7. Precursor signals ──
    signals = []
    if density_risk > 60:
        signals.append("High local density detected")
    if velocity_risk > 50:
        signals.append("Velocity direction conflict in crowd")
    if counterflow_risk > 30:
        signals.append("Counter-flow detected")
    if anomaly_risk > 50:
        signals.append("Multiple anomalous trajectories")

    # ── 8. Flag individuals ──
    flagged_ids = []
    for p in pedestrian_states:
        reasons = []
        if p.anomaly_score > 0.25:
            reasons.append(f"unusual movement (score {p.anomaly_score:.2f})")
        if p.divergence_score > 0.50:
            reasons.append(f"unexpected behaviour (divergence {p.divergence_score:.2f})")
        speed = np.linalg.norm(p.velocity)
        if speed > 3.0:
            reasons.append(f"abnormal speed ({speed:.1f} m/s)")
        if reasons:
            p.is_flagged   = True
            p.flag_reasons = reasons
            flagged_ids.append(p.ped_id)

    return CrowdRiskReport(
        overall_risk      = round(overall_risk, 1),
        density_risk      = round(density_risk, 1),
        velocity_risk     = round(velocity_risk, 1),
        anomaly_risk      = round(anomaly_risk, 1),
        flagged_ids       = flagged_ids,
        risk_label        = label,
        precursor_signals = signals,
    )


# ─── Batch scoring utility ────────────────────────────────────────────────────

def build_pedestrian_states(obs_np, pred_np, ped_ids):
    """
    Convenience function to build PedestrianState list from numpy arrays.
    obs_np  : (N, 8,  2)
    pred_np : (N, 12, 2)
    ped_ids : (N,)
    """
    states = []
    for i, pid in enumerate(ped_ids):
        vel = obs_np[i, -1, :] - obs_np[i, -2, :]   # last displacement as velocity
        states.append(PedestrianState(
            ped_id    = int(pid),
            positions = obs_np[i],
            predicted = pred_np[i],
            velocity  = vel,
            speed     = float(np.linalg.norm(vel)),
        ))
    return states