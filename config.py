"""Shared configuration for training and inference.

Keeping a single source of truth here fixes the old train/inference drift
(e.g. checkpoint name mismatch, hardcoded num_classes/window in several files).
Values can still be overridden on the command line - see build_argparser().
"""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # data
    train_dir: str = "./emg_data/train"
    test_dir: str = "./emg_data/test"
    val_dir: str = ""              # optional explicit validation dir; if empty, carve from train
    val_fraction: float = 0.2      # per-file holdout from train_dir for model selection
    window_size: int = 64          # timesteps per window
    channel: int = 1               # EMG channels (features) per timestep
    num_classes: int = 4
    step_train: int = 1            # sliding-window stride for training
    step_eval: int = 64            # non-overlapping windows for evaluation

    # model
    depth: int = 4                 # num_layers of the deep bidirectional GRU

    # optim
    optimizer: str = "SGD"         # SGD | Adam | RMSProp
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-5
    batch_size: int = 32
    epochs: int = 300
    amp: bool = False              # mixed precision (CUDA only)

    # class imbalance: weight the loss by inverse class frequency
    class_weighting: bool = True

    # training loop behaviour
    early_stop_patience: int = 20  # stop if val macro-F1 does not improve
    seed: int = 42
    deterministic: bool = False    # cudnn.deterministic + benchmark=False

    # dataloader
    num_workers: int = 0           # map-style + in-memory: workers add little
    pin_memory: bool = True

    # io
    out_dir: str = "./timestep_64"
    ckpt_name: str = "model.pt"    # single name shared by train + inference
    resume: bool = False           # resume training from ckpt if present

    @property
    def ckpt_path(self) -> str:
        import os
        return os.path.join(self.out_dir, self.ckpt_name)


def build_argparser() -> argparse.ArgumentParser:
    c = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=c.train_dir)
    p.add_argument("--test-dir", default=c.test_dir)
    p.add_argument("--val-dir", default=c.val_dir)
    p.add_argument("--val-fraction", type=float, default=c.val_fraction)
    p.add_argument("--window-size", type=int, default=c.window_size)
    p.add_argument("--channel", type=int, default=c.channel)
    p.add_argument("--num-classes", type=int, default=c.num_classes)
    p.add_argument("--step-train", type=int, default=c.step_train)
    p.add_argument("--step-eval", type=int, default=c.step_eval)
    p.add_argument("--depth", type=int, default=c.depth)
    p.add_argument("--optimizer", default=c.optimizer, choices=["SGD", "Adam", "RMSProp"])
    p.add_argument("--learning-rate", type=float, default=c.learning_rate)
    p.add_argument("--momentum", type=float, default=c.momentum)
    p.add_argument("--weight-decay", type=float, default=c.weight_decay)
    p.add_argument("--batch-size", type=int, default=c.batch_size)
    p.add_argument("--epochs", type=int, default=c.epochs)
    p.add_argument("--amp", action="store_true", default=c.amp)
    p.add_argument("--no-class-weighting", dest="class_weighting", action="store_false", default=c.class_weighting)
    p.add_argument("--early-stop-patience", type=int, default=c.early_stop_patience)
    p.add_argument("--seed", type=int, default=c.seed)
    p.add_argument("--deterministic", action="store_true", default=c.deterministic)
    p.add_argument("--num-workers", type=int, default=c.num_workers)
    p.add_argument("--out-dir", default=c.out_dir)
    p.add_argument("--ckpt-name", default=c.ckpt_name)
    p.add_argument("--resume", action="store_true", help="resume training from ckpt if present")
    return p


def config_from_args(args) -> Config:
    return Config(
        train_dir=args.train_dir, test_dir=args.test_dir,
        val_dir=args.val_dir, val_fraction=args.val_fraction,
        window_size=args.window_size, channel=args.channel, num_classes=args.num_classes,
        step_train=args.step_train, step_eval=args.step_eval, depth=args.depth,
        optimizer=args.optimizer, learning_rate=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs,
        amp=args.amp, class_weighting=args.class_weighting,
        early_stop_patience=args.early_stop_patience, seed=args.seed,
        deterministic=args.deterministic, num_workers=args.num_workers,
        out_dir=args.out_dir, ckpt_name=args.ckpt_name, resume=args.resume,
    )
