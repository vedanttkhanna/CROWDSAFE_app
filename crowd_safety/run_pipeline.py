"""
Run the full vision pipeline on a video file or webcam.

Usage:
    python run_pipeline.py --video input.mp4
    python run_pipeline.py --video input.mp4 --output annotated.mp4
    python run_pipeline.py --webcam
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   type=str, default=None)
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--webcam",  action="store_true")
    parser.add_argument("--scene",   type=str, default="eth")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    processor = VideoProcessor()
    processor.load_models(CKPT_DIR, scene=args.scene)

    if args.webcam:
        print("Starting webcam...")
        processor.process_video(0, args.output, show=not args.no_show)
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Video not found: {args.video}")
            return
        print(f"Processing: {args.video}")
        processor.process_video(args.video, args.output, show=not args.no_show)
    else:
        print("Provide --video path or --webcam flag")
        print("Example: python run_pipeline.py --video crowd.mp4 --output out.mp4")


if __name__ == "__main__":
    main()