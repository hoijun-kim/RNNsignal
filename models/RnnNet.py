import torch
import torch.nn as nn


class Model(nn.Module):
    """Per-timestep GRU sequence labeller.

    Input : (batch, time_slot, channel)
    Output: (batch, time_slot, num_class) raw logits  (softmax applied by the loss / at inference)
    """

    def __init__(self, time_slot: int = 64, depth: int = 4, num_class: int = 4, channel: int = 1):
        super().__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel

        self.GRU1 = nn.GRU(input_size=channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        # NOTE: batch_first=True on BOTH GRUs. The original code had this GRU set to
        # batch_first=False while it consumed GRU1's batch-first output, which made the
        # recurrence run over the batch axis instead of time. Keep both batch_first=True.
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=time_slot // 2, batch_first=True,
                           num_layers=depth, bidirectional=True)
        # GRU5 is bidirectional with hidden_size=time_slot//2 -> output feature dim = time_slot.
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.Dense64 = nn.Linear(1000, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)

    def forward(self, x) -> torch.Tensor:
        x, _ = self.GRU1(x)          # (B, T, time_slot)
        x_gru, _ = self.GRU5(x)      # (B, T, time_slot)
        x_den = self.Dense1000(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense64(x_den)
        logits = self.CLS(x_den)     # (B, T, num_class) -- raw logits
        return logits


if __name__ == '__main__':
    from torchinfo import summary

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(time_slot=64, depth=4, num_class=4, channel=1).to(device)
    a = torch.ones([2, 64, 1], device=device)  # (batch, time_slot, channel)
    out = model(a)
    print(f"out shape: {tuple(out.shape)}")     # expect (2, 64, 4)
    assert out.shape == (2, 64, 4)
    summary(model, input_size=(32, 64, 1), depth=4)
