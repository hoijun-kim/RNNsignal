# RNNsignal

Per-timestep classification of surface EMG (electromyography) signals with recurrent networks.
A sliding window of raw EMG is labelled sample-by-sample into one of N movement classes - the
sequence-labelling core used in my biosignal / prosthesis-control research.

## What it does

Given a continuous EMG recording, the model assigns a class to **every timestep**, not just to
the window as a whole: a 64-sample window in produces 64 per-sample predictions out. This fits
online control, where the intended action has to be decoded continuously rather than once per
gesture.

- Input: raw EMG, shape `(batch, window=64, channels=1)`
- Output: per-timestep class **logits**, shape `(batch, 64, 4)` (softmax/argmax applied by the loss / at inference)
- Defaults: 1 channel, window 64, 4 classes

## Models

| file | model | idea |
|------|-------|------|
| `models/RnnNet.py` | GRU stack (default) | 1-layer GRU encoder -> 4-layer **bidirectional** GRU -> MLP head -> per-timestep logits |
| `models/ClickNet.py` | LSTM | window -> last hidden state -> MLP -> one label per window (whole-window variant) |

`RnnNet.Model` is the network used by `train.py` and `inference.py`.

## Data format

Plain CSV files under `emg_data/train/` and `emg_data/test/`, one recording per file. Each row is:

    ch0[, ch1, ...], label

- the first `channel` columns are the EMG sample(s)
- the last column is an integer class id (`0 .. num_classes-1`)

The loader (`datasets/Dataset.py`) is a plain map-style `torch.utils.data.Dataset`. It slides a
`window_size` window with a given `step` **within each file** (windows never straddle two
recordings), materializes the windows once, and returns integer per-timestep labels. Inputs are
z-score standardized using per-channel statistics computed on the **training** set only (the stats
are saved in the checkpoint and reapplied at inference). The recordings themselves are not committed.

## Setup

    pip install -r requirements.txt

Versions are pinned to a verified-working set (Python 3.13, CPU torch). For a CUDA build, install
torch from the appropriate `download.pytorch.org/whl/cuXXX` index. CUDA, Apple MPS, and CPU are all
detected automatically. (The old `torchdata` datapipe loader has been removed - no `torchdata` dependency.)

## Train

Put your recordings in `emg_data/train` (used for training; the validation split is carved from
here) and `emg_data/test` (held out for the final `inference.py` report), then:

    python train.py

Defaults live in `config.py`; every knob is also a CLI flag (`python train.py --help`) - e.g.
`--epochs`, `--batch-size`, `--optimizer`, `--learning-rate`, `--depth`, `--step-train`,
`--num-classes`, `--no-class-weighting`, `--deterministic`, `--resume`, `--val-dir`, `--val-fraction`.

The loop uses **class-weighted `CrossEntropyLoss`** (handles the heavy `rest`-class imbalance),
seeds all RNGs, and **selects the best model by validation macro-F1**. The validation set is kept
**disjoint from `emg_data/test`**: by default a per-recording fraction (`--val-fraction`, 0.2) is held
out from the training files (or pass an explicit `--val-dir`), so `emg_data/test` is reserved for the
final `inference.py` report and the reported test metrics stay unbiased. The best model is written to
`timestep_64/model.pt` (plus `last.pt` and `train_log.csv`); early stopping and `--resume` are supported.

## Inference

    python inference.py

Loads `timestep_64/model.pt` (weights + normalization stats + model config), runs the test set in
**eval mode with autograd disabled**, and reports accuracy, macro-F1, and a confusion matrix. It
also measures true per-window forward latency (warmup + CUDA sync, timing only the forward pass).
Use `--show-predictions` to print predicted vs ground-truth classes per window.

## Layout

    config.py            shared config + CLI flags (single source of truth)
    models/RnnNet.py     GRU sequence-labelling model (default), returns logits
    models/ClickNet.py   LSTM whole-window classifier (alternative)
    datasets/Dataset.py  map-style windowing dataset (per-file, cached, integer labels)
    datasets/utils.py    CSV reading, normalization stats, class weights
    train.py             training loop: class-weighted CE, per-epoch val, best-checkpoint, early stop
    inference.py         checkpoint evaluation: acc / macro-F1 / confusion + true latency
    null_test.py         data sanity check (class distribution)
    utils/               small helpers

## Notes

This is research code shared to document the approach. Configuration has sensible defaults and can
be overridden via CLI flags; dataset paths and the checkpoint name default from `config.py`.

Part of my biosignal ML work - see [github.com/hoijun-kim](https://github.com/hoijun-kim).
