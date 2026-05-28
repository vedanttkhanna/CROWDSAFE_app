"""
Day 1 Run Script
Usage:
    python run_day1.py                  # full 100-epoch run
    python run_day1.py --epochs 20      # quick test
    python run_day1.py --scene hotel
"""

import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.trainer import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--scene",      type=str, default="eth",
                        choices=["eth","hotel","univ","zara1","zara2"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience",   type=int, default=15)
    parser.add_argument("--models",     nargs="+",
                        default=["vanilla","bilstm","stacked"])
    args = parser.parse_args()

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    RESULTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results.json")

    print("\n" + "="*60)
    print(f"  Models: {args.models}  |  Scene: {args.scene}  |  Epochs: {args.epochs}")
    print("="*60)

    all_results = []
    for model_name in args.models:
        r = train_model(
            model_name=model_name, data_dir=DATA_DIR,
            test_scene=args.scene, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
            ckpt_dir=CKPT_DIR, results_path=RESULTS,
        )
        all_results.append(r)

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  {'Model':<14} {'ADE':>8} {'FDE':>8} {'Params':>10}")
    print(f"  {'-'*44}")
    for r in all_results:
        print(f"  {r['model']:<14} {r['test_ade']:>8.4f} {r['test_fde']:>8.4f} {r['params']:>10,}")

    best = min(all_results, key=lambda r: r["test_ade"])
    print(f"\n  Best: {best['model'].upper()} — ADE {best['test_ade']:.4f}m\n")

if __name__ == "__main__":
    main()