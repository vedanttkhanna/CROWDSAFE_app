
"""
Training Engine — shared for all Day 1 models
Metrics: ADE (Average Displacement Error), FDE (Final Displacement Error)
Both in metres. Lower is better.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import get_loaders
from models.day1_models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ade(pred, target):
    return torch.norm(pred - target, dim=-1).mean().item()

def fde(pred, target):
    return torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1).mean().item()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for obs, pred_gt in loader:
        obs, pred_gt = obs.to(DEVICE), pred_gt.to(DEVICE)
        optimizer.zero_grad()
        pred = model(obs)
        loss = criterion(pred, pred_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ade_sum, fde_sum, count = 0.0, 0.0, 0
    for obs, pred_gt in loader:
        obs, pred_gt = obs.to(DEVICE), pred_gt.to(DEVICE)
        pred = model(obs)
        ade_sum += ade(pred, pred_gt) * obs.size(0)
        fde_sum += fde(pred, pred_gt) * obs.size(0)
        count   += obs.size(0)
    return ade_sum / count, fde_sum / count


def train_model(model_name, data_dir, test_scene="eth", epochs=100,
                lr=1e-3, batch_size=64, patience=15,
                ckpt_dir="checkpoints", results_path="outputs/results.json"):

    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}  |  Test scene: {test_scene}  |  Device: {DEVICE}")
    print(f"{'='*55}")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(data_dir, test_scene, batch_size)

    model     = get_model(model_name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    criterion = nn.MSELoss()

    best_val_ade = float("inf")
    patience_ctr = 0
    history      = {"train_loss": [], "val_ade": [], "val_fde": []}
    t0           = time.time()

    for epoch in range(1, epochs + 1):
        train_loss       = train_one_epoch(model, train_loader, optimizer, criterion)
        val_ade, val_fde = evaluate(model, val_loader)
        scheduler.step(val_ade)

        history["train_loss"].append(round(train_loss, 6))
        history["val_ade"].append(round(val_ade, 4))
        history["val_fde"].append(round(val_fde, 4))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} "
                  f"| val ADE {val_ade:.4f} | val FDE {val_fde:.4f} "
                  f"| {time.time()-t0:.0f}s")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"{model_name}_{test_scene}_best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"{model_name}_{test_scene}_best.pt"),
        map_location=DEVICE, weights_only=True
    ))
    test_ade, test_fde = evaluate(model, test_loader)
    params = sum(p.numel() for p in model.parameters())

    result = {
        "model": model_name, "test_scene": test_scene,
        "test_ade": round(test_ade, 4), "test_fde": round(test_fde, 4),
        "best_val_ade": round(best_val_ade, 4),
        "params": params, "train_time_s": round(time.time() - t0, 1),
        "history": history,
    }

    print(f"\n  ✓ {model_name} | TEST ADE: {test_ade:.4f} | TEST FDE: {test_fde:.4f}")

    all_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            content = f.read().strip()
            if content:
                all_results = json.load(f) if False else json.loads(content)
    all_results = [r for r in all_results
                   if not (r["model"] == model_name and r["test_scene"] == test_scene)]
    all_results.append(result)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return result
