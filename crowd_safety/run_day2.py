"""
Day 2 Run Script
Trains Social LSTM, Attention LSTM, LSTM Autoencoder, Seq2Seq LSTM.
Then runs anomaly scoring demo on test data.

Usage:
    python run_day2.py
    python run_day2.py --epochs 20 --patience 8
    python run_day2.py --models social attention
    python run_day2.py --skip_training   (anomaly demo only)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer_day2 import train_model_day2, DEVICE
from data.dataset import get_loaders
from models.day2_models import get_model
from utils.anomaly import (
    score_reconstruction, score_divergence,
    build_pedestrian_states, compute_risk_score
)


def run_anomaly_demo(data_dir, ckpt_dir, test_scene="eth"):
    print("\n" + "="*55)
    print("  ANOMALY SCORING DEMO")
    print("="*55)

    _, _, test_loader = get_loaders(data_dir, test_scene, batch_size=32)
    obs_batch, pred_batch = next(iter(test_loader))
    obs_batch  = obs_batch.to(DEVICE)
    pred_batch = pred_batch.to(DEVICE)

    # Load autoencoder
    ae_path = os.path.join(ckpt_dir, f"autoencoder_{test_scene}_best.pt")
    if not os.path.exists(ae_path):
        print("  Autoencoder checkpoint not found — skipping anomaly demo")
        return

    autoencoder = get_model("autoencoder").to(DEVICE)
    autoencoder.load_state_dict(torch.load(ae_path, map_location=DEVICE,
                                            weights_only=True))

    # Load seq2seq
    seq_path = os.path.join(ckpt_dir, f"seq2seq_{test_scene}_best.pt")
    seq2seq  = get_model("seq2seq").to(DEVICE)
    if os.path.exists(seq_path):
        seq2seq.load_state_dict(torch.load(seq_path, map_location=DEVICE,
                                            weights_only=True))

    # Score
    ae_scores, ae_flags       = score_reconstruction(autoencoder, obs_batch)
    div_scores, div_flags     = score_divergence(seq2seq, obs_batch, pred_batch)

    # Build pedestrian states
    obs_np  = obs_batch.cpu().numpy()
    pred_np = pred_batch.cpu().numpy()
    ped_ids = list(range(len(obs_np)))
    states  = build_pedestrian_states(obs_np, pred_np, ped_ids)

    for i, state in enumerate(states):
        state.anomaly_score   = float(ae_scores[i])
        state.divergence_score = float(div_scores[i])

    # Compute crowd risk
    report = compute_risk_score(states)

    print(f"\n  Batch size          : {len(states)} pedestrians")
    print(f"  Flagged individuals : {len(report.flagged_ids)}")
    print(f"  Flagged IDs         : {report.flagged_ids[:10]}")
    print(f"\n  Risk Scores:")
    print(f"    Overall    : {report.overall_risk:5.1f} / 100  [{report.risk_label}]")
    print(f"    Density    : {report.density_risk:5.1f} / 100")
    print(f"    Velocity   : {report.velocity_risk:5.1f} / 100")
    print(f"    Anomaly    : {report.anomaly_risk:5.1f} / 100")

    if report.precursor_signals:
        print(f"\n  Precursor signals detected:")
        for sig in report.precursor_signals:
            print(f"    ⚠  {sig}")

    # Show top 3 most anomalous individuals
    sorted_states = sorted(states, key=lambda s: s.anomaly_score, reverse=True)
    print(f"\n  Top 3 most anomalous pedestrians:")
    for s in sorted_states[:3]:
        print(f"    Ped {s.ped_id:3d} | anomaly: {s.anomaly_score:.3f} "
              f"| divergence: {s.divergence_score:.3f} "
              f"| flagged: {s.is_flagged} "
              f"| reasons: {s.flag_reasons}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--scene",         type=str, default="eth",
                        choices=["eth","hotel","univ","zara1","zara2"])
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--patience",      type=int, default=15)
    parser.add_argument("--models",        nargs="+",
                        default=["social","attention","autoencoder","seq2seq"])
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS  = os.path.join(BASE_DIR, "outputs", "results.json")

    if not args.skip_training:
        print("\n" + "="*60)
        print(f"  DAY 2 TRAINING")
        print(f"  Models : {args.models}")
        print(f"  Scene  : {args.scene}  |  Epochs: {args.epochs}")
        print("="*60)

        all_results = []
        for model_name in args.models:
            r = train_model_day2(
                model_name   = model_name,
                data_dir     = DATA_DIR,
                test_scene   = args.scene,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                patience     = args.patience,
                ckpt_dir     = CKPT_DIR,
                results_path = RESULTS,
            )
            all_results.append(r)

        print("\n" + "="*60)
        print("  DAY 2 RESULTS")
        print("="*60)
        print(f"  {'Model':<14} {'ADE/Err':>9} {'FDE':>8} {'Params':>10}")
        print(f"  {'-'*45}")
        for r in all_results:
            print(f"  {r['model']:<14} {r['test_ade']:>9.4f} "
                  f"{r['test_fde']:>8.4f} {r['params']:>10,}")

    run_anomaly_demo(DATA_DIR, CKPT_DIR, args.scene)


if __name__ == "__main__":
    main()