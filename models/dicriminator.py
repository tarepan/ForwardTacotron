import torch
import torch.nn.functional as F
from models.tacotron import CBHG


class Discriminator(torch.nn.Module):

    def __init__(self, n_mels: int, lstm_dim: int) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(
            n_mels + 512, lstm_dim, batch_first=True, bidirectional=True)
        self.postnet = CBHG(K=8,
                            in_channels=2 * lstm_dim,
                            channels=512,
                            proj_channels=[512, 2 * lstm_dim],
                            num_highways=4)
        self.lin = torch.nn.Linear(2 * lstm_dim, 1)

    def forward(self, x, x_out):
        x = x.transpose(1, 2)
        x = torch.cat([x, x_out], dim=-1)
        x_feat_1, _ = self.rnn(x)
        x = x_feat_1.transpose_(1, 2)
        x_feat_2 = self.postnet(x)
        x = F.dropout(x_feat_2, p=0.1, training=True)
        x = self.lin(x)
        return x.squeeze(), x_feat_1.squeeze().detach(), x_feat_2.squeeze().detach()
