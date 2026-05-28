"""
Video Processor — wires together the full pipeline:
YOLOv8 detection → DeepSORT tracking → perspective transform
→ trajectory buffer → LSTM inference → anomaly scoring → annotated output
"""

import cv2
import torch
import numpy as np
import time
from collections import defaultdict

from pipeline.detector import CrowdTracker
from pipeline.transform import PerspectiveTransform
from pipeline.trajectory_buffer import TrajectoryBuffer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.day1_models import get_model as get_day1_model
from models.day2_models import get_model as get_day2_model
from utils.anomaly import (
    build_pedestrian_states, compute_risk_score,
    score_reconstruction
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {
    "Normal":   (0,   200, 0  ),
    "Monitor":  (0,   200, 255),
    "Warning":  (0,   140, 255),
    "CRITICAL": (0,   0,   255),
}


class VideoProcessor:
    def __init__(self):
        self.tracker           = None
        self.transform         = PerspectiveTransform()
        self.buffer            = TrajectoryBuffer()
        self.models            = {}
        self.autoencoder       = None
        self.seq2seq           = None
        self.anomaly_scores    = defaultdict(float)
        self.divergence_scores = defaultdict(float)
        self.flagged_ids       = set()

    def load_models(self, ckpt_dir, scene="eth"):
        print(f"Loading models from {ckpt_dir}...")

        for name in ["vanilla", "bilstm", "stacked"]:
            path = os.path.join(ckpt_dir, f"{name}_{scene}_best.pt")
            if os.path.exists(path):
                m = get_day1_model(name).to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE,
                                             weights_only=True))
                m.eval()
                self.models[name] = m
                print(f"  ✓ {name}")

        ae_path = os.path.join(ckpt_dir, f"autoencoder_{scene}_best.pt")
        if os.path.exists(ae_path):
            self.autoencoder = get_day2_model("autoencoder").to(DEVICE)
            self.autoencoder.load_state_dict(
                torch.load(ae_path, map_location=DEVICE, weights_only=True)
            )
            self.autoencoder.eval()
            print(f"  ✓ autoencoder")

        seq_path = os.path.join(ckpt_dir, f"seq2seq_{scene}_best.pt")
        if os.path.exists(seq_path):
            self.seq2seq = get_day2_model("seq2seq").to(DEVICE)
            self.seq2seq.load_state_dict(
                torch.load(seq_path, map_location=DEVICE, weights_only=True)
            )
            self.seq2seq.eval()
            print(f"  ✓ seq2seq")

        print(f"Loaded {len(self.models) + 2} models.")

    def _run_lstm_inference(self, ready_tracks):
        if not ready_tracks:
            return {}, {}

        ids    = list(ready_tracks.keys())
        obs_np = np.stack([ready_tracks[i] for i in ids])
        obs_t  = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)

        predictions = {}
        if "bilstm" in self.models:
            with torch.no_grad():
                pred = self.models["bilstm"](obs_t).cpu().numpy()
            for i, tid in enumerate(ids):
                predictions[tid] = pred[i]

        ae_scores = {}
        if self.autoencoder is not None:
            scores, _ = score_reconstruction(
                self.autoencoder, obs_t, threshold=0.35
            )
            for i, tid in enumerate(ids):
                prev = self.anomaly_scores[tid]
                self.anomaly_scores[tid] = 0.7 * prev + 0.3 * float(scores[i])
                ae_scores[tid] = self.anomaly_scores[tid]

        return predictions, ae_scores

    def _annotate_frame(self, frame, tracks, predictions, ae_scores, risk_report):
        annotated = frame.copy()

        # ── Per-person bounding boxes ─────────────────────────────────────────
        for track in tracks:
            tid             = track["track_id"]
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            cx              = int((x1 + x2) / 2)
            cy              = int((y1 + y2) / 2)
            is_flagged      = tid in self.flagged_ids
            score           = ae_scores.get(tid, 0.0)
            color           = (0, 0, 255) if is_flagged else (0, 200, 0)
            thickness       = 3 if is_flagged else 1

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label with black backdrop
            label       = f"ID:{tid}" + (f" RISK:{score:.2f}" if is_flagged else "")
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 10),
                          (x1 + lw + 4, y1),
                          (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Pulsing ring for flagged
            if is_flagged:
                radius = int((x2 - x1) * 0.7)
                cv2.circle(annotated, (cx, cy), radius, (0, 0, 255), 2)

        # ── Risk overlay — transparent backdrop ───────────────────────────────
        lines = [
            (f"CROWD RISK: {risk_report.risk_label}", 0.6, 2, True),
            (f"Score    : {risk_report.overall_risk:.0f}/100",  0.5, 1, False),
            (f"Density  : {risk_report.density_risk:.0f}/100",  0.5, 1, False),
            (f"Flagged  : {len(risk_report.flagged_ids)} person(s)", 0.5, 1, False),
            (f"Tracking : {len(tracks)} people",                0.5, 1, False),
        ]
        for sig in risk_report.precursor_signals[:3]:
            lines.append((f"! {sig}", 0.4, 1, False))

        box_h = 14 + len(lines) * 22
        box_w = 310

        # Semi-transparent black box
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Thin border
        cv2.rectangle(annotated, (8, 8), (8 + box_w, 8 + box_h), (40, 40, 40), 1)

        # Text lines — y_pos increments every line without exception
        risk_color = COLORS.get(risk_report.risk_label, (0, 200, 0))
        y_pos      = 30

        for i, (text, scale, thick, is_title) in enumerate(lines):
            if is_title:
                color = risk_color
            elif 'Flagged' in text and risk_report.flagged_ids:
                color = (0, 0, 255)
            else:
                color = (180, 180, 180)

            cv2.putText(annotated, text, (16, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y_pos += 22   # always increment — was the bug

        return annotated

    def process_video(self, input_path, output_path=None, show=True):
        if self.tracker is None:
            self.tracker = CrowdTracker()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.transform.calibrate_eth(width, height)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        fps_timer   = time.time()
        print(f"\nProcessing video — press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            tracks     = self.tracker.process_frame(frame)
            active_ids = {t["track_id"] for t in tracks}
            self.buffer.clear_lost_tracks(active_ids)

            for track in tracks:
                tid    = track["track_id"]
                cx, cy = track["centre"]
                wx, wy = self.transform.pixel_to_world(cx, cy)
                self.buffer.update(tid, wx, wy)

            ready            = self.buffer.get_ready_tracks(min_len=4)
            preds, ae_scores = self._run_lstm_inference(ready)

            states = []
            for track in tracks:
                tid = track["track_id"]
                if tid in ready:
                    obs = ready[tid]
                    vel = obs[-1] - obs[-2] if len(obs) >= 2 else np.zeros(2)
                    from utils.anomaly import PedestrianState
                    states.append(PedestrianState(
                        ped_id        = tid,
                        positions     = obs,
                        predicted     = preds.get(tid, np.zeros((12, 2))),
                        velocity      = vel,
                        speed         = float(np.linalg.norm(vel)),
                        anomaly_score = ae_scores.get(tid, 0.0),
                    ))

            risk_report      = compute_risk_score(states)
            self.flagged_ids = set(risk_report.flagged_ids)
            annotated        = self._annotate_frame(
                frame, tracks, preds, ae_scores, risk_report
            )

            if frame_count % 30 == 0:
                elapsed    = time.time() - fps_timer
                actual_fps = 30 / elapsed
                fps_timer  = time.time()
                print(f"  Frame {frame_count} | {actual_fps:.1f} FPS | "
                      f"{len(tracks)} tracked | {len(self.flagged_ids)} flagged | "
                      f"Risk: {risk_report.risk_label} ({risk_report.overall_risk:.0f})")

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Crowd Safety System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_count} frames.")