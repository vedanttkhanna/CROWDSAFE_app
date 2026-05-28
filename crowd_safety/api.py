"""
FastAPI — Crowd Safety System
Serves all 5 trained models + anomaly engine as REST endpoints.

Endpoints:
    GET  /                        health check
    GET  /models                  list all models + ADE/FDE scores
    POST /predict                 trajectory prediction from raw coords
    POST /anomaly                 anomaly score for a trajectory
    POST /video/frame             process single video frame → annotated image + risk report
    GET  /risk/demo               run demo on ETH test data
    WS   /ws/video                websocket for live video streaming
"""

import os
import sys
import json
import base64
import asyncio
from unittest import result
import numpy as np
import torch
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.day1_models import get_model as get_day1_model
from models.day2_models import get_model as get_day2_model
from utils.anomaly import (
    score_reconstruction, build_pedestrian_states,
    compute_risk_score, PedestrianState
)
from data.dataset import get_loaders

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS  = os.path.join(BASE_DIR, "outputs", "results.json")

app = FastAPI(
    title="Crowd Safety System",
    description="Multi-LSTM pedestrian trajectory prediction and anomaly detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Model registry ───────────────────────────────────────────────────────────

MODELS      = {}
AUTOENCODER = None
SEQ2SEQ     = None


def load_all_models(scene="eth"):
    global AUTOENCODER, SEQ2SEQ

    print(f"\nLoading models (scene={scene}, device={DEVICE})...")

    for name in ["vanilla", "bilstm", "stacked"]:
        path = os.path.join(CKPT_DIR, f"{name}_{scene}_best.pt")
        if os.path.exists(path):
            m = get_day1_model(name).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE,
                                          weights_only=True))
            m.eval()
            MODELS[name] = m
            print(f"  ✓ {name}")

    ae_path = os.path.join(CKPT_DIR, f"autoencoder_{scene}_best.pt")
    if os.path.exists(ae_path):
        AUTOENCODER = get_day2_model("autoencoder").to(DEVICE)
        AUTOENCODER.load_state_dict(torch.load(ae_path, map_location=DEVICE,
                                                weights_only=True))
        AUTOENCODER.eval()
        print(f"  ✓ autoencoder")

    seq_path = os.path.join(CKPT_DIR, f"seq2seq_{scene}_best.pt")
    if os.path.exists(seq_path):
        SEQ2SEQ = get_day2_model("seq2seq").to(DEVICE)
        SEQ2SEQ.load_state_dict(torch.load(seq_path, map_location=DEVICE,
                                            weights_only=True))
        SEQ2SEQ.eval()
        print(f"  ✓ seq2seq")

    print(f"Loaded {len(MODELS) + 2} models.\n")


# Load on startup
load_all_models()


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TrajectoryInput(BaseModel):
    # 8 observed positions as flat list [[x,y], [x,y], ...]
    observed: List[List[float]]
    ped_id:   Optional[int] = 0

class BatchTrajectoryInput(BaseModel):
    trajectories: List[TrajectoryInput]

class FrameInput(BaseModel):
    # Base64 encoded JPEG/PNG frame
    frame_b64: str
    scene:     Optional[str] = "eth"


# ─── Helper functions ─────────────────────────────────────────────────────────

def obs_to_tensor(observed: List[List[float]]) -> torch.Tensor:
    arr = np.array(observed, dtype=np.float32)   # (8, 2)
    if arr.shape != (8, 2):
        raise HTTPException(400, f"Expected 8 positions, got {arr.shape[0]}")
    # Normalize to relative coords
    arr = arr - arr[0]
    return torch.tensor(arr).unsqueeze(0).to(DEVICE)  # (1, 8, 2)


def tensor_to_list(t: torch.Tensor) -> list:
    return t.squeeze(0).cpu().tolist()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":  "running",
        "models":  list(MODELS.keys()) + ["autoencoder", "seq2seq"],
        "device":  str(DEVICE),
    }


@app.get("/models")
def list_models():
    """Returns all model metadata + benchmark results."""
    results = []

    # Load saved results
    saved = {}
    if os.path.exists(RESULTS):
        try:
            with open(RESULTS, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                if content:
                    for r in json.loads(content):
                        saved[r["model"]] = r
        except Exception:
            pass

    model_info = {
        "vanilla": {
            "name":        "Vanilla LSTM",
            "description": "Single-layer encoder-decoder. Baseline model.",
            "type":        "prediction",
            "trained":     True,
        },
        "bilstm": {
            "name":        "Bidirectional LSTM",
            "description": "Reads trajectory forward and backward. Best for direction-change detection.",
            "type":        "prediction",
            "trained":     True,
        },
        "stacked": {
            "name":        "Stacked LSTM",
            "description": "2-layer deep LSTM. Lower layer: step dynamics. Upper layer: movement intent.",
            "type":        "prediction",
            "trained":     True,
        },
        "social": {
            "name":        "Social LSTM",
            "description": "Grid-based neighbour pooling. O(N²) — too slow for CPU training.",
            "type":        "prediction",
            "trained":     False,
            "paper_ade":   0.73,
            "paper_fde":   1.47,
            "note":        "Alahi et al. CVPR 2016",
        },
        "attention": {
            "name":        "Attention LSTM",
            "description": "Learned attention weights per neighbour. Visualizable influence graph.",
            "type":        "prediction",
            "trained":     False,
            "note":        "Per-sample attention loop too slow on CPU",
        },
        "autoencoder": {
            "name":        "LSTM Autoencoder",
            "description": "Trained on normal trajectories. High reconstruction error = anomaly.",
            "type":        "anomaly",
            "trained":     True,
        },
        "seq2seq": {
            "name":        "Seq2Seq LSTM",
            "description": "Encoder-decoder. Prediction divergence from actual = risk signal.",
            "type":        "anomaly",
            "trained":     True,
        },
    }

    for key, info in model_info.items():
        if key in saved:
            info["test_ade"] = saved[key]["test_ade"]
            info["test_fde"] = saved[key]["test_fde"]
            info["params"]   = saved[key]["params"]
        results.append({"id": key, **info})

    return {"models": results}


@app.post("/predict")
def predict(data: TrajectoryInput):
    """
    Predict next 12 positions from 8 observed positions.
    Returns predictions from all trained models.
    """
    obs_t = obs_to_tensor(data.observed)

    predictions = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            pred = model(obs_t)
            predictions[name] = tensor_to_list(pred)

    return {
        "ped_id":      data.ped_id,
        "observed":    data.observed,
        "predictions": predictions,
    }


@app.post("/anomaly")
def anomaly(data: TrajectoryInput):
    """
    Score a single trajectory for anomalous behaviour.
    Returns reconstruction error and flag status.
    """
    if AUTOENCODER is None:
        raise HTTPException(503, "Autoencoder not loaded")

    obs_t  = obs_to_tensor(data.observed)
    scores, flags = score_reconstruction(AUTOENCODER, obs_t, threshold=0.25)

    score    = float(scores[0])
    flagged  = bool(flags[0])
    reasons  = []
    if score > 0.25:
        reasons.append(f"High reconstruction error ({score:.3f})")

    # Speed anomaly check
    obs_np = np.array(data.observed)
    speeds = np.linalg.norm(np.diff(obs_np, axis=0), axis=1) / 0.4
    if speeds.max() > 3.0:
        reasons.append(f"Abnormal speed ({speeds.max():.1f} m/s)")
    if speeds.max() < 0.1:
        reasons.append("Stationary in crowd")

    return {
        "ped_id":          data.ped_id,
        "anomaly_score":   round(score, 4),
        "flagged":         flagged,
        "threshold":       0.25,
        "reasons":         reasons,
        "risk_level":      "HIGH" if score > 0.5 else "MEDIUM" if score > 0.25 else "LOW",
    }


@app.post("/batch/anomaly")
def batch_anomaly(data: BatchTrajectoryInput):
    """Score multiple pedestrians at once — used by the frontend."""
    if AUTOENCODER is None:
        raise HTTPException(503, "Autoencoder not loaded")

    results = []
    for traj in data.trajectories:
        result = anomaly(traj)
        results.append(result)

    flagged_count = sum(1 for r in results if r["flagged"])
    avg_score     = np.mean([r["anomaly_score"] for r in results])

    return {
        "total":         len(results),
        "flagged_count": flagged_count,
        "avg_score":     round(float(avg_score), 4),
        "results":       results,
    }


@app.post("/video/frame")
async def process_frame(data: FrameInput):
    """
    Process a single video frame.
    Input : base64 encoded image
    Output: annotated image (base64) + risk report + flagged IDs
    """
    try:
        img_bytes = base64.b64decode(data.frame_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Lazy-load video processor
    if not hasattr(app.state, "processor"):
        from pipeline.video_processor import VideoProcessor
        app.state.processor = VideoProcessor()
        app.state.processor.load_models(CKPT_DIR)
        app.state.processor.transform.calibrate_eth(
            frame.shape[1], frame.shape[0]
        )
        if app.state.processor.tracker is None:
            from pipeline.detector import CrowdTracker
            app.state.processor.tracker = CrowdTracker()

    proc = app.state.processor

    # Run pipeline
    tracks  = proc.tracker.process_frame(frame)
    active  = {t["track_id"] for t in tracks}
    proc.buffer.clear_lost_tracks(active)

    for track in tracks:
        cx, cy = track["centre"]
        wx, wy = proc.transform.pixel_to_world(cx, cy)
        proc.buffer.update(track["track_id"], wx, wy)

    ready         = proc.buffer.get_ready_tracks()
    preds, scores = proc._run_lstm_inference(ready)

    states = []
    for track in tracks:
        tid = track["track_id"]
        if tid in ready:
            obs = ready[tid]
            vel = obs[-1] - obs[-2]
            states.append(PedestrianState(
                ped_id        = tid,
                positions     = obs,
                predicted     = preds.get(tid, np.zeros((12, 2))),
                velocity      = vel,
                speed         = float(np.linalg.norm(vel)),
                anomaly_score = scores.get(tid, 0.0),
            ))

    risk_report = compute_risk_score(states)
    proc.flagged_ids = set(risk_report.flagged_ids)

    annotated = proc._annotate_frame(frame, tracks, preds, scores, risk_report)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "frame_b64":    frame_b64,
        "risk": {
            "overall":   risk_report.overall_risk,
            "density":   risk_report.density_risk,
            "velocity":  risk_report.velocity_risk,
            "anomaly":   risk_report.anomaly_risk,
            "label":     risk_report.risk_label,
            "flagged":   risk_report.flagged_ids,
            "signals":   risk_report.precursor_signals,
        },
        "tracks": [
            {
                "id":      t["track_id"],
                "bbox":    t["bbox"],
                "centre":  list(t["centre"]),
                "flagged": t["track_id"] in proc.flagged_ids,
                "score":   round(scores.get(t["track_id"], 0.0), 4),
            }
            for t in tracks
        ],
    }


@app.get("/risk/demo")
def risk_demo():
    """
    Run anomaly scoring on a batch from ETH test set.
    No video needed — uses pre-extracted coordinates.
    Good for testing the frontend without a video file.
    """
    if AUTOENCODER is None:
        raise HTTPException(503, "Autoencoder not loaded")

    _, _, test_loader = get_loaders(DATA_DIR, test_scene="eth", batch_size=32)
    obs_batch, pred_batch = next(iter(test_loader))
    obs_t  = obs_batch.to(DEVICE)
    pred_t = pred_batch.to(DEVICE)

    scores, flags = score_reconstruction(AUTOENCODER, obs_t, threshold=0.25)
    obs_np  = obs_batch.numpy()
    pred_np = pred_batch.numpy()

    states = build_pedestrian_states(obs_np, pred_np, list(range(len(obs_np))))
    for i, state in enumerate(states):
        state.anomaly_score = float(scores[i])

    risk = compute_risk_score(states)

    # Get predictions from all models
    predictions_by_model = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            pred = model(obs_t[:5])
            predictions_by_model[name] = pred.cpu().tolist()

    return {
        "total_pedestrians": len(states),
        "flagged_ids":       risk.flagged_ids,
        "flagged_count":     len(risk.flagged_ids),
        "risk": {
            "overall":  risk.overall_risk,
            "density":  risk.density_risk,
            "velocity": risk.velocity_risk,
            "anomaly":  risk.anomaly_risk,
            "label":    risk.risk_label,
            "signals":  risk.precursor_signals,
        },
        "sample_trajectories": [
            {
                "ped_id":        i,
                "observed":      obs_np[i].tolist(),
                "anomaly_score": round(float(scores[i]), 4),
                "flagged":       bool(flags[i]),
            }
            for i in range(min(10, len(obs_np)))
        ],
        "model_predictions": predictions_by_model,
    }


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket for live video streaming from browser.
    Client sends base64 frames, server sends back annotated frames + risk data.
    """
    await websocket.accept()
    print("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            if payload.get("type") == "frame":
                frame_input = FrameInput(frame_b64=payload["frame_b64"])
                result = await process_frame(frame_input)
                await websocket.send_text(json.dumps({
                    "type":       "result",
                    "frame_b64":  result["frame_b64"],
                    "risk":       result["risk"],
                    "tracks":     result["tracks"],
                }))
            elif payload.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)