import torch


class Discriminator(torch.nn.Module):

    def __init__(self, n_mels: int, lstm_dim: int) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(
            n_mels + 512, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, 1)

    def forward(self, x, x_out):
        x = x.transpose(1, 2)
        x = torch.cat([x, x_out], dim=-1)
        x_feat, _ = self.rnn(x)
        x = self.lin(x_feat)
        #x = torch.sigmoid(x)
        return x.squeeze(), x_feat.squeeze()
