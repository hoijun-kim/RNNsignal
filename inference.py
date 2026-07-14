"""Evaluate a trained checkpoint on the test set.

Loads the checkpoint written by train.py (weights + normalization stats + model config),
runs the model in eval mode with autograd disabled, reports accuracy / macro-F1 / confusion
matrix, and measures true per-window forward latency (warmup + CUDA sync, timing the forward
pass only - not the print/formatting overhead the old version accidentally measured).
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix

from config import Config
from datasets.Dataset import EMGWindowDataset
from models.RnnNet import Model
from train import get_device


def parse_args():
    c = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=c.ckpt_path)
    p.add_argument("--test-dir", default=c.test_dir)
    p.add_argument("--step-eval", type=int, default=None,
                   help="eval window stride; defaults to the value stored in the checkpoint")
    p.add_argument("--batch-size", type=int, default=1)   # batch=1 for per-window latency
    p.add_argument("--show-predictions", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device: {device}")

    ck = torch.load(args.ckpt, map_location=device)
    mc = ck["config"]
    mean = np.asarray(ck["mean"], dtype=np.float32)
    std = np.asarray(ck["std"], dtype=np.float32)
    print(f"loaded {args.ckpt} (trained {ck['epoch'] + 1} epochs, best_f1={ck.get('best_f1', float('nan')):.4f})")

    step_eval = args.step_eval if args.step_eval is not None else mc.get("step_eval", 64)
    test_ds = EMGWindowDataset(args.test_dir, mc["window_size"], mc["channel"], mc["num_classes"],
                               step=step_eval, mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"test windows: {len(test_ds)}")

    model = Model(mc["window_size"], mc["depth"], mc["num_classes"], mc["channel"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()  # <- disables Dropout(0.5); without this predictions are random at inference

    nc = mc["num_classes"]
    acc = MulticlassAccuracy(num_classes=nc, average="micro").to(device)  # overall (top-line) accuracy
    f1 = MulticlassF1Score(num_classes=nc, average="macro").to(device)
    cm = MulticlassConfusionMatrix(num_classes=nc).to(device)

    # warmup so lazy init / first-call overhead does not pollute the latency stats
    with torch.inference_mode():
        warm = next(iter(test_loader))["data"].to(device)
        for _ in range(3):
            model(warm)
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies = []
    with torch.inference_mode():
        for feats in test_loader:
            data, target = feats["data"].to(device), feats["label"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(data)               # <- time ONLY the forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1e3)  # ms

            preds = logits.argmax(dim=-1)
            acc.update(preds.reshape(-1), target.reshape(-1))
            f1.update(preds.reshape(-1), target.reshape(-1))
            cm.update(preds.reshape(-1), target.reshape(-1))
            if args.show_predictions:
                print("pred:", preds[0].tolist())
                print("gt  :", target[0].tolist())
                print("=" * 40)

    lat = np.array(latencies)
    print(f"\naccuracy : {acc.compute().item():.4f}")
    print(f"macro-F1 : {f1.compute().item():.4f}")
    print("confusion matrix (rows=gt, cols=pred):")
    print(cm.compute().cpu().numpy())
    print(f"\nper-window latency: mean {lat.mean():.3f} ms | p50 {np.percentile(lat, 50):.3f} | "
          f"p95 {np.percentile(lat, 95):.3f} ms  (batch={args.batch_size}, n={len(lat)})")


if __name__ == "__main__":
    main()
