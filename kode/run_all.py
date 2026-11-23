#!/usr/bin/env python3
"""
Master single-command reproduction script for UTS Deep Learning:
"Hands-on Deep Learning: Classification, Regression, and Clustering"

Usage:
  python kode/run_all.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission
  python kode/run_all.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission --eval_only
"""
import argparse
import os
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Reproduce all tasks for UTS Deep Learning")
    parser.add_argument("--data_dir", type=str, default=r"E:\Code\data_uts_dl", help="Path to data_uts_dl directory")
    parser.add_argument("--out_dir", type=str, default=r"E:\Code\uts_dl_submission", help="Path to submission root directory")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate saved checkpoints without retraining")
    parser.add_argument("--task", type=int, choices=[0, 1, 2, 3], default=0, help="Run specific task (0 = all)")
    args = parser.parse_args()

    kode_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = []
    if args.task in (0, 1):
        scripts.append(("Task 1 (Multimodal Classification)", os.path.join(kode_dir, "task1_classification.py")))
    if args.task in (0, 2):
        scripts.append(("Task 2 (Time-Series Regression - RUL)", os.path.join(kode_dir, "task2_rul.py")))
    if args.task in (0, 3):
        scripts.append(("Task 3 (Multimodal Clustering)", os.path.join(kode_dir, "task3_clustering.py")))

    t0 = time.time()
    for name, script_path in scripts:
        print("=" * 72)
        print(f"RUNNING: {name}")
        print("=" * 72)
        cmd = [sys.executable, script_path, "--data_dir", args.data_dir, "--out_dir", args.out_dir]
        if args.eval_only:
            cmd.append("--eval_only")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] {name} exited with code {res.returncode}")
            sys.exit(res.returncode)

    elapsed = time.time() - t0
    print("=" * 72)
    print(f"ALL TASKS COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
    print("=" * 72)

if __name__ == "__main__":
    main()
