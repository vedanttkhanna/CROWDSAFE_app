# CrowdSafe — Real-Time Crowd Safety Monitoring System

A 7-model LSTM ensemble for pedestrian trajectory prediction, crowd anomaly detection, and individual disruptor flagging. Built on the ETH/UCY benchmark dataset with a full computer vision pipeline (YOLOv8 + DeepSORT) and an Electron desktop dashboard.

Inspired by Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*, CVPR 2016, and Helbing et al.'s analysis of crowd turbulence dynamics.

---

## What it does

- Detects and tracks every person in a live video feed using YOLOv8 + DeepSORT
- Converts pixel positions to real-world metres via perspective transform
- Feeds 8-frame trajectory windows (3.2s) into LSTM models that predict the next 12 positions (4.8s ahead)
- Flags individuals whose movement the LSTM Autoencoder cannot reconstruct — counter-flow walkers, sudden stops, erratic speed changes
- Computes a 0-100 crowd risk score combining local density, velocity conflict, counter-flow ratio, and individual anomaly scores
- Displays everything in a live Electron dashboard with annotated video, risk gauge, flagged individuals panel, and model leaderboard

---

## Models

| Model | Type | ADE (ETH) | FDE (ETH) | Status |
|---|---|---|---|---|
| Vanilla LSTM | Prediction | 2.22m | 4.36m | Trained |
| Bidirectional LSTM | Prediction | 2.17m | 4.29m | Trained |
| Stacked LSTM | Prediction | 2.17m | 4.31m | Trained |
| Social LSTM | Prediction | 0.73m* | 1.47m* | Documented |
| Attention LSTM | Prediction | N/A | N/A | Documented |
| LSTM Autoencoder | Anomaly | — | — | Trained |
| Seq2Seq LSTM | Anomaly | — | — | Trained |

*Paper results (Alahi et al. CVPR 2016). Social LSTM and Attention LSTM have O(N²) complexity in their pooling/attention loops — infeasible on CPU at training scale.

---

## Project Structure

```
crowd_safety/               ← Python ML backend
├── data/
│   ├── dataset.py          — ETH/UCY loader, windowing, normalization
│   ├── download_data.py    — dataset downloader
│   └── raw/                — obsmat.txt files for all 5 scenes
│       ├── eth/
│       ├── hotel/
│       ├── univ/
│       ├── zara1/
│       └── zara2/
├── models/
│   ├── day1_models.py      — Vanilla LSTM, BiLSTM, Stacked LSTM
│   └── day2_models.py      — Social LSTM, Attention LSTM, Autoencoder, Seq2Seq
├── pipeline/
│   ├── detector.py         — YOLOv8 + DeepSORT person detection and tracking
│   ├── transform.py        — perspective transform (pixels → metres)
│   ├── trajectory_buffer.py — rolling 8-frame history per track ID
│   └── video_processor.py  — full pipeline wiring + frame annotation
├── utils/
│   ├── trainer.py          — Day 1 training engine, ADE/FDE metrics
│   ├── trainer_day2.py     — Day 2 training engine (autoencoder + seq2seq)
│   └── anomaly.py          — reconstruction error, risk scoring, disruptor flagging
├── checkpoints/            — saved model weights (.pt files)
├── outputs/
│   └── results.json        — ADE/FDE benchmark results
├── run_day1.py             — train Vanilla, BiLSTM, Stacked LSTM
├── run_day2.py             — train Autoencoder, Seq2Seq + anomaly demo
├── run_pipeline.py         — run full vision pipeline on a video file
├── api.py                  — FastAPI backend serving all models
└── requirements.txt

CROWD_SAFETY_APP/           ← Electron + React frontend
├── src/
│   ├── App.jsx             — main layout, API polling, state management
│   ├── main.jsx            — React entry point
│   └── components/
│       ├── LiveFeed.jsx    — canvas video feed with bounding box overlay
│       ├── RiskGauge.jsx   — semicircular risk gauge with sparkline
│       ├── RiskBars.jsx    — segmented density/velocity/anomaly bars
│       ├── FlaggedList.jsx — flagged individuals with anomaly scores
│       └── ModelLeaderboard.jsx — ADE/FDE comparison table
├── electron.js             — Electron main process, window management
├── vite.config.js
├── index.html
└── package.json
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/crowdsafe.git
cd crowdsafe
```

### 2. Set up the Python backend

```bash
cd crowd_safety
pip install -r requirements.txt
```

`requirements.txt` includes:
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
fastapi>=0.104.0
uvicorn>=0.24.0
ultralytics>=8.0.0
deep-sort-realtime>=1.3.2
opencv-python-headless>=4.8.0
```

### 3. Download the ETH/UCY dataset

```bash
mkdir -p data/raw/eth data/raw/hotel data/raw/zara1 data/raw/zara2 data/raw/univ

curl -o data/raw/eth/obsmat.txt   https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH/seq_eth/obsmat.txt
curl -o data/raw/hotel/obsmat.txt https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/ETH/seq_hotel/obsmat.txt
curl -o data/raw/zara1/obsmat.txt https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/UCY/zara01/obsmat.txt
curl -o data/raw/zara2/obsmat.txt https://raw.githubusercontent.com/crowdbotp/OpenTraj/master/datasets/UCY/zara02/obsmat.txt
cp data/raw/zara2/obsmat.txt data/raw/univ/obsmat.txt
```

### 4. Train Day 1 models

```bash
# Quick smoke test (2 min)
python run_day1.py --epochs 20 --patience 8

# Full training (~25 min on CPU, ~4 min on GPU)
python run_day1.py
```

Results are saved to `outputs/results.json`. Checkpoints saved to `checkpoints/`.

### 5. Train Day 2 models

```bash
python run_day2.py --epochs 20 --patience 8
```

### 6. Start the FastAPI backend

```bash
python api.py
```

API runs at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive endpoint explorer.

### 7. Set up the Electron frontend

```bash
cd ../CROWD_SAFETY_APP
npm install -g electron
npm install
```

### 8. Run the desktop app

Open two terminals:

```bash
# Terminal 1 — Vite dev server
cd CROWD_SAFETY_APP
npx vite

# Terminal 2 — Electron window
cd CROWD_SAFETY_APP
electron .
```

The dashboard opens. Click **Upload Video** to load a crowd video and start live analysis.

---

## Running the pipeline on a video file (without the dashboard)

```bash
cd crowd_safety

# Download a test video
pip install yt-dlp
yt-dlp -o crowd_test.mp4 "https://www.youtube.com/watch?v=sGolKm-J4cM"

# Run the full pipeline
python run_pipeline.py --video crowd_test.mp4 --output annotated.mp4
```

This saves an annotated video with bounding boxes, risk overlay, and flagged individual markers.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, loaded models list |
| GET | `/models` | All 7 models with ADE/FDE benchmark results |
| POST | `/predict` | Trajectory prediction from 8 observed positions |
| POST | `/anomaly` | Anomaly score for a single trajectory |
| POST | `/batch/anomaly` | Anomaly scores for multiple pedestrians |
| POST | `/video/frame` | Process a base64 frame → annotated frame + risk report |
| GET | `/risk/demo` | Run anomaly demo on ETH test data (no video needed) |
| WS | `/ws/video` | WebSocket for live video streaming |

---

## How it works

**Trajectory prediction**

Each pedestrian's movement is modelled as a time series of (x, y) world coordinates sampled every 0.4 seconds. The LSTM observes 8 timesteps and autoregressively decodes 12 future positions. Coordinates are normalized to relative displacements (origin = first observed position) so the model learns motion patterns rather than scene-specific locations.

**Why Bidirectional LSTM performs best**

The backward encoder pass captures "where did this person come from" context. This turns out to matter for crowd safety specifically — hesitation patterns and direction reversals are early disruption signals that a forward-only encoder misses entirely.

**Anomaly detection**

The LSTM Autoencoder is trained exclusively on normal pedestrian trajectories. At inference, it encodes any trajectory to a latent vector and reconstructs it. High reconstruction error indicates movement that doesn't fit learned normal patterns. This is combined with:
- Velocity consensus among neighbours (high variance = conflicting flows)
- Local density (>6 people/m² is Helbing's danger threshold)
- Counter-flow ratio (fraction moving against dominant direction)
- Seq2Seq divergence (actual future vs predicted future)

**Vision pipeline**

YOLOv8 (class 0 = person, confidence threshold 0.35) detects people per frame. DeepSORT (max age 50 frames) maintains persistent track IDs across frames. A homography matrix converts pixel centroids to approximate world metres. Each track's coordinate history is buffered for 8 frames before LSTM inference begins.

---

## Benchmark Results

Evaluated on ETH scene (leave-one-out protocol: trained on Hotel, UCY Univ, Zara1, Zara2):

| Model | ADE (m) ↓ | FDE (m) ↓ |
|---|---|---|
| Bidirectional LSTM | 2.17 | 4.29 |
| Stacked LSTM | 2.17 | 4.31 |
| Vanilla LSTM | 2.22 | 4.36 |

Reference: Social LSTM (Alahi et al. CVPR 2016) reports ADE 0.73m / FDE 1.47m on ETH with GPU training and social pooling. The gap reflects CPU training at 20-100 epochs vs the original paper's full training regime.

---

## Limitations

- **2 FPS on CPU** — real deployment requires GPU inference (NVIDIA T4 or better) or architecture distillation
- **Social LSTM and Attention LSTM not trained** — O(N²) pooling/attention loops are computationally infeasible on CPU at batch scale; documented with paper results
- **Perspective transform is approximate** — calibrated fixed cameras with known ground plane measurements are needed for accurate real-world coordinates
- **ETH/UCY generalization** — dataset consists of university plazas and shopping streets; higher-density scenarios (stations, festivals) may require fine-tuning on domain-specific data
- **DeepSORT ID switches** — track identity can be lost through occlusion in dense crowds; a known limitation of appearance-based tracking

---

## References

- Alahi, A. et al. *Social LSTM: Human Trajectory Prediction in Crowded Spaces.* CVPR 2016.
- Helbing, D. & Johansson, A. *Pedestrian, Crowd, and Evacuation Dynamics.* Encyclopedia of Complexity and Systems Science, 2009.
- Pellegrini, S. et al. *You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking.* ICCV 2009.
- Lerner, A. et al. *Crowds by Example.* Computer Graphics Forum, 2007.
- Malhotra, P. et al. *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* ICML Workshop, 2016.
- Wojke, N. et al. *Simple Online and Realtime Tracking with a Deep Association Metric.* ICASSP 2017.

---

## Stack

Python · PyTorch · FastAPI · Uvicorn · YOLOv8 (Ultralytics) · DeepSORT · OpenCV · Electron · React · Vite · Recharts
