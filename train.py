"""Train the per-timestep GRU EMG classifier.

Zero-arg ``python train.py`` works with the defaults in config.py; every knob is also a
CLI flag (``python train.py --help``). Weights + normalization stats + config are saved to
one checkpoint that inference.py reads back.
"""
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from tqdm import tqdm

from config import build_argparser, config_from_args
from datasets.Dataset import EMGWindowDataset
from datasets.utils import compute_norm_stats, compute_class_weights, list_csv_files
from models.RnnNet import Model
from utils.AddFunc import check_folder


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(name, params, lr, momentum, weight_decay):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "RMSProp":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r} (expected SGD | Adam | RMSProp)")


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)  # overall (top-line) accuracy
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    total_loss, n = 0.0, 0
    for feats in loader:
        data, target = feats["data"].to(device), feats["label"].to(device)
        logits = model(data)                                   # (B, W, C)
        loss = criterion(logits.reshape(-1, num_classes), target.reshape(-1))
        preds = logits.argmax(dim=-1)
        acc.update(preds.reshape(-1), target.reshape(-1))
        f1.update(preds.reshape(-1), target.reshape(-1))
        cm.update(preds.reshape(-1), target.reshape(-1))
        total_loss += loss.item() * data.size(0)
        n += data.size(0)
    return {"loss": total_loss / max(n, 1), "acc": acc.compute().item(),
            "macro_f1": f1.compute().item(), "cm": cm.compute().cpu().numpy()}


def main():
    cfg = config_from_args(build_argparser().parse_args())
    set_seed(cfg.seed, cfg.deterministic)
    check_folder(cfg.out_dir)
    device = get_device()
    print(f"device: {device}")

    # Validation set for model selection / early stopping is kept DISJOINT from cfg.test_dir
    # (which is reserved for the final inference.py report). Either use an explicit --val-dir,
    # or hold out a per-file fraction of the training recordings.
    all_train_files = list_csv_files(cfg.train_dir)
    if cfg.val_dir:
        train_files = all_train_files
        val_files = list_csv_files(cfg.val_dir)
    elif len(all_train_files) > 1 and cfg.val_fraction > 0:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(all_train_files))
        n_val = max(1, int(round(len(all_train_files) * cfg.val_fraction)))
        n_val = min(n_val, len(all_train_files) - 1)  # keep >=1 training file
        val_idx = set(perm[:n_val].tolist())
        train_files = [f for i, f in enumerate(all_train_files) if i not in val_idx]
        val_files = [all_train_files[i] for i in sorted(val_idx)]
    else:
        # too few files to split: fall back to test_dir as val (selection is then test-biased)
        print("WARNING: cannot carve a validation split from train_dir; using test_dir for "
              "model selection (reported test metrics will be selection-biased).")
        train_files, val_files = all_train_files, list_csv_files(cfg.test_dir)
    print(f"train files: {len(train_files)} | val files: {len(val_files)}")

    # normalization stats + class weights are derived from the TRAIN split only.
    mean, std = compute_norm_stats(train_files, cfg.channel)
    print(f"norm mean={mean.tolist()} std={std.tolist()}")

    train_ds = EMGWindowDataset(files=train_files, window_size=cfg.window_size, channel=cfg.channel,
                                num_classes=cfg.num_classes, step=cfg.step_train, mean=mean, std=std)
    val_ds = EMGWindowDataset(files=val_files, window_size=cfg.window_size, channel=cfg.channel,
                              num_classes=cfg.num_classes, step=cfg.step_eval, mean=mean, std=std)
    print(f"train windows: {len(train_ds)} | val windows: {len(val_ds)}")

    use_pin = cfg.pin_memory and device.type == "cuda"  # pin_memory only helps for CUDA transfers
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=use_pin,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=use_pin)

    if cfg.class_weighting:
        weights = torch.from_numpy(compute_class_weights(train_ds.labels.reshape(-1), cfg.num_classes)).to(device)
        print(f"class weights: {weights.tolist()}")
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = Model(cfg.window_size, cfg.depth, cfg.num_classes, cfg.channel).to(device)
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), cfg.learning_rate,
                               cfg.momentum, cfg.weight_decay)
    scaler = amp.GradScaler(device.type, enabled=cfg.amp)
    train_acc = MulticlassAccuracy(num_classes=cfg.num_classes, average="micro").to(device)

    start_epoch, best_f1, bad_epochs = 0, -1.0, 0
    resume_path = os.path.join(cfg.out_dir, "last.pt")  # latest epoch state, not the best-only model
    if cfg.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_f1 = ck.get("best_f1", -1.0)
        bad_epochs = ck.get("bad_epochs", 0)  # restore early-stop patience state across resume
        print(f"resumed from {resume_path} at epoch {start_epoch} (best_f1={best_f1:.4f}, bad_epochs={bad_epochs})")

    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    new_log = (not os.path.exists(log_path)) or os.path.getsize(log_path) == 0
    log_f = open(log_path, "a", newline="")
    logw = csv.writer(log_f)
    if new_log:
        logw.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "best_f1"])

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_acc.reset()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for feats in pbar:
            data, target = feats["data"].to(device), feats["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=cfg.amp):
                logits = model(data)                                    # (B, W, C)
                loss = criterion(logits.reshape(-1, cfg.num_classes), target.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                train_acc.update(logits.argmax(dim=-1).reshape(-1), target.reshape(-1))
            running_loss += loss.item() * data.size(0)
            seen += data.size(0)
            pbar.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{train_acc.compute().item():.4f}")

        tr_loss = running_loss / max(seen, 1)
        tr_acc = train_acc.compute().item()
        val = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, bad_epochs = val["macro_f1"], 0
        else:
            bad_epochs += 1

        print(f"[{epoch}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.4f} macroF1 {val['macro_f1']:.4f}"
              f"{'  <-- best' if improved else ''}")
        logw.writerow([epoch, tr_loss, tr_acc, val["loss"], val["acc"], val["macro_f1"], best_f1])
        log_f.flush()

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_f1": best_f1, "bad_epochs": bad_epochs,
                # store as plain lists so torch.load(weights_only=True) (the 2.6+ default) accepts them
                "mean": mean.tolist(), "std": std.tolist(),
                "config": {"window_size": cfg.window_size, "channel": cfg.channel,
                           "num_classes": cfg.num_classes, "depth": cfg.depth,
                           "step_eval": cfg.step_eval}}
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))
        if improved:
            torch.save(ckpt, cfg.ckpt_path)  # best model -> the name inference loads
            print("confusion matrix (rows=gt, cols=pred):")
            print(val["cm"])

        if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
            print(f"early stop at epoch {epoch} (no val macro-F1 gain for {bad_epochs} epochs)")
            break

    log_f.close()
    print(f"done. best val macro-F1 = {best_f1:.4f}. best model at {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
