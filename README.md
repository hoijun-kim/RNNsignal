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
- Output: per-timestep class probabilities, shape `(batch, 64, 4)`
- Defaults: 1 channel, window 64, 4 classes

## Models

| file | model | idea |
|------|-------|------|
| `models/RnnNet.py` | GRU stack (default) | 1-layer GRU encoder -> 4-layer **bidirectional** GRU -> MLP head -> per-timestep softmax |
| `models/ClickNet.py` | LSTM | window -> last hidden state -> MLP -> one label per window (whole-window variant) |

`RnnNet.Model` is the network used by `train.py` and `inference.py`.

## Data format

Plain CSV files under `emg_data/train/` and `emg_data/test/`, one recording per file. Each row is:

    ch0[, ch1, ...], label

- the first `channel` columns are the EMG sample(s)
- the last column is an integer class id (`0 .. num_classes-1`)

The loader (`datasets/Dataset.py`, built on `torchdata` pipes) slides a `window_size` window with
a given `step` and one-hot-encodes the label window. The recordings themselves are not committed.

## Setup

    pip install -r requirements.txt

CUDA, Apple MPS, and CPU are all detected automatically.

## Train

Put CSVs in `emg_data/train` and `emg_data/test`, then:

    python train.py

Hyper-parameters live at the top of `train.py` (`depth`, `batch`, `time_slot`, `channel`,
`num_classes`, optimiser, `learning_rate`, `end_epochs`, ...). Trained weights are written to
`result3.pt`.

## Inference

    python inference.py

Loads a checkpoint (`result2.pt` by default), runs the test set with a non-overlapping window
(`step=64`), and prints predicted vs ground-truth classes per window together with per-window
latency.

## Layout

    models/RnnNet.py     GRU sequence-labelling model (default)
    models/ClickNet.py   LSTM whole-window classifier (alternative)
    datasets/Dataset.py  torchdata pipeline: CSV -> rolling window -> one-hot
    datasets/utils.py    row parsing / file filter
    train.py             training loop (config at top of file)
    inference.py         checkpoint evaluation + latency print
    null_test.py         data sanity check (fraction of zero / rest labels)
    utils/               small helpers

## Notes

This is research code: configuration is edited in the files rather than passed as CLI flags, and
dataset paths and checkpoint names are hard-coded. It is shared to document the approach, not as a
turnkey package.

Part of my biosignal ML work - see [github.com/hoijun-kim](https://github.com/hoijun-kim).
