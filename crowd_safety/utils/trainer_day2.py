"""
Day 2 Training Engine
Handles prediction models (Social, Attention, Seq2Seq)
and the Autoencoder separately since it has a different training objective.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import get_loaders
from models.day2_models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ade(pred, target):
    return torch.norm(pred - target, dim=-1).mean().item()

def fde(pred, target):
    return torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1).mean().item()


# ─── Prediction model training ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, model_name):
    model.train()
    total_loss = 0.0
    for obs, pred_gt in loader:
        obs, pred_gt = obs.to(DEVICE), pred_gt.to(DEVICE)
        optimizer.zero_grad()
        if model_name == "seq2seq":
            pred = model(obs, pred_gt, teacher_forcing=0.5)
        else:
            pred = model(obs)
        loss = criterion(pred, pred_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, model_name):
    model.eval()
    ade_sum, fde_sum, count = 0.0, 0.0, 0
    for obs, pred_gt in loader:
        obs, pred_gt = obs.to(DEVICE), pred_gt.to(DEVICE)
        if model_name == "seq2seq":
            pred = model(obs)
        else:
            pred = model(obs)
        ade_sum += ade(pred, pred_gt) * obs.size(0)
        fde_sum += fde(pred, pred_gt) * obs.size(0)
        count   += obs.size(0)
    return ade_sum / count, fde_sum / count


# ─── Autoencoder training ─────────────────────────────────────────────────────

def train_autoencoder_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    criterion  = nn.MSELoss()
    for obs, _ in loader:
        obs = obs.to(DEVICE)
        optimizer.zero_grad()
        recon, _ = model(obs)
        loss     = criterion(recon, obs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_autoencoder(model, loader):
    model.eval()
    total_error = 0.0
    count       = 0
    for obs, _ in loader:
        obs    = obs.to(DEVICE)
        recon, _ = model(obs)
        error  = torch.norm(recon - obs, dim=-1).mean()
        total_error += error.item() * obs.size(0)
        count       += obs.size(0)
    return total_error / count


# ─── Unified training function ────────────────────────────────────────────────

def train_model_day2(
    model_name,
    data_dir,
    test_scene    = "eth",
    epochs        = 100,
    lr            = 1e-3,
    batch_size    = 64,
    patience      = 15,
    ckpt_dir      = "checkpoints",
    results_path  = "outputs/results.json",
):
    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}  |  Scene: {test_scene}  |  Device: {DEVICE}")
    print(f"{'='*55}")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        data_dir, test_scene, batch_size
    )

    model     = get_model(model_name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val  = float("inf")
    patience_ctr = 0
    history   = {"train_loss": [], "val_metric": []}
    t0        = time.time()
    is_ae     = (model_name == "autoencoder")

    for epoch in range(1, epochs + 1):
        if is_ae:
            train_loss = train_autoencoder_epoch(model, train_loader, optimizer)
            val_metric = evaluate_autoencoder(model, val_loader)
            metric_name = "recon_err"
        else:
            train_loss         = train_one_epoch(model, train_loader, optimizer,
                                                  criterion, model_name)
            val_metric, val_fde = evaluate(model, val_loader, model_name)
            metric_name = "val_ADE"

        scheduler.step(val_metric)
        history["train_loss"].append(round(train_loss, 6))
        history["val_metric"].append(round(val_metric, 4))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} "
                  f"| {metric_name} {val_metric:.4f} | {time.time()-t0:.0f}s")

        if val_metric < best_val:
            best_val     = val_metric
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"{model_name}_{test_scene}_best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"{model_name}_{test_scene}_best.pt"),
        map_location=DEVICE, weights_only=True
    ))

    if is_ae:
        test_metric = evaluate_autoencoder(model, test_loader)
        test_ade    = test_metric
        test_fde    = 0.0
        print(f"\n  ✓ {model_name} | TEST recon_err: {test_metric:.4f}")
    else:
        test_ade, test_fde = evaluate(model, test_loader, model_name)
        print(f"\n  ✓ {model_name} | TEST ADE: {test_ade:.4f} | TEST FDE: {test_fde:.4f}")

    params = sum(p.numel() for p in model.parameters())
    result = {
        "model": model_name, "test_scene": test_scene,
        "test_ade": round(test_ade, 4), "test_fde": round(test_fde, 4),
        "best_val": round(best_val, 4), "params": params,
        "train_time_s": round(time.time() - t0, 1),
        "history": history,
    }

    all_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8-sig') as f:
                content = f.read().strip()
                if content:
                    all_results = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            all_results = []

    all_results = [r for r in all_results
                   if not (r["model"] == model_name and r["test_scene"] == test_scene)]
    all_results.append(result)
    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    return result