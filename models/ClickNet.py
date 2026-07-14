import torch
import torch.nn as nn


class ClickNet(nn.Module):
    """Alternative whole-window LSTM classifier: window -> last hidden state -> one label."""

    def __init__(self, n_features, n_hidden, n_sequence, n_layers, n_classes):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_sequence = n_sequence
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear_1 = nn.Linear(in_features=n_hidden, out_features=128)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.linear_2 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x) -> torch.Tensor:
        # let nn.LSTM allocate zero initial state on the right device/dtype (no manual hidden)
        out, _ = self.lstm(x.view(len(x), self.n_sequence, -1))
        out = out[:, -1, :]           # last timestep
        out = self.linear_1(out)
        out = self.dropout_1(out)
        out = self.linear_2(out)
        return out


if __name__ == '__main__':
    from torchinfo import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features, n_hidden, n_sequence, n_layers, n_classes = 3, 64, 64, 2, 4
    model = ClickNet(n_features, n_hidden, n_sequence, n_layers, n_classes).to(device)
    a = torch.ones([2, n_sequence, n_features], device=device)
    out = model(a)
    print(f"out shape: {tuple(out.shape)}")   # expect (2, 4)
    summary(model, input_size=(32, n_sequence, n_features), depth=4)
