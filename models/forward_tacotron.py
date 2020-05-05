from pathlib import Path
from typing import Union

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from models.tacotron import CBHG


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        return self.expand(x, dur)
     
    @staticmethod
    def build_index(duration, x):
        duration[duration < 0] = 0
        tot_duration = duration.cumsum(1).detach().cpu().numpy().astype('int')
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return torch.LongTensor(index).to(duration.device)

    def expand(self, x, dur):
        idx = self.build_index(dur, x)
        y = torch.gather(x, 1, idx)
        return y


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class DurationPredictor(nn.Module):

    def __init__(self, in_dims, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_chars,
                 durpred_conv_dims,
                 durpred_rnn_dims,
                 durpred_dropout,
                 rnn_dim,
                 prenet_k,
                 prenet_dims,
                 postnet_k,
                 postnet_dims,
                 highways,
                 dropout,
                 n_mels):

        super().__init__()
        self.rnn_dim = rnn_dim
        self.n_mels = n_mels
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(embed_dims,
                                          conv_dims=durpred_conv_dims,
                                          rnn_dims=durpred_rnn_dims,
                                          dropout=durpred_dropout)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
                           num_highways=highways)
        self.rnn = nn.GRU(rnn_dim,
                            rnn_dim,
                            batch_first=True,
                            bidirectional=False)
        self.pre_net = PreNet(n_mels, fc2_dims=n_mels)
        self.lin = torch.nn.Linear(rnn_dim, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=highways)
        self.dropout = dropout
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)
        self.I = nn.Linear(2 * prenet_dims + n_mels, rnn_dim)

    def forward(self, x, mel, dur):
        device = next(self.parameters()).device  # use same device as parameters

        if self.training:
            self.step += 1

        x = self.embedding(x)
        dur_hat = self.dur_pred(x)
        dur_hat = dur_hat.squeeze()

        x = x.transpose(1, 2)
        x = self.prenet(x)
        x = self.lr(x, dur)

        bsize = mel.size(0)
        start_mel = torch.zeros(bsize, 1, self.n_mels, device=device)
        h = torch.zeros(1, bsize, self.rnn_dim, device=device)

        mel = mel.transpose(1, 2)
        mel = torch.cat([start_mel, mel[:, :-1, :]], dim=1)
        x = self.pad(x, mel.size(1))
        mel = self.pre_net(mel)
        x = torch.cat([x, mel], dim=-1)
        x = self.I(x)

        x, _ = self.rnn(x, h)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        #x_post = self.pad(x_post, mel.size(1))
        return x, x_post, dur_hat

    def forward_2(self, x, mels, dur):
        if self.training:
            self.step += 1

        device = next(self.parameters()).device  # use same device as parameters

        x = self.embedding(x)
        dur_hat = self.dur_pred(x)
        dur_hat = dur_hat.squeeze()

        x = x.transpose(1, 2)
        x = self.prenet(x)
        x = self.lr(x, dur)
        x = self.pad(x, mels.size(2))

        b_size = x.size(0)
        h = torch.zeros(b_size, self.rnn_dim, device=device)
        mel = torch.zeros(b_size, self.n_mels, device=device)
        rnn = self.get_gru_cell(self.rnn)
        out_mels = []

        for i in range(x.size(1)):
            x_t = x[:, i, :]
            x_t = torch.cat([x_t, mel], dim=-1)
            x_t = self.I(x_t)

            h = rnn(x_t, h)
            x_t = F.dropout(h,
                            p=self.dropout,
                            training=self.training)
            x_t = self.lin(x_t)
            out_mels.append(x_t.unsqueeze(1))
            mel = x_t

        x = torch.cat(out_mels, dim=1)
        x = x.transpose(1, 2)
        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return x, x_post, dur_hat

    def generate(self, x, alpha=1.0):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        x = self.embedding(x)
        dur = self.dur_pred(x, alpha=alpha)
        dur = dur.squeeze(2)

        x = x.transpose(1, 2)
        x = self.prenet(x)
        x = self.lr(x, dur)

        b_size = x.size(0)
        h = torch.zeros(b_size, self.rnn_dim, device=device)
        mel = torch.zeros(b_size, self.n_mels, device=device)
        rnn = self.get_gru_cell(self.rnn)
        out_mels = []

        for i in range(x.size(1)):
            mel = self.pre_net(mel)
            x_t = x[:, i, :]
            x_t = torch.cat([x_t, mel], dim=-1)
            x_t = self.I(x_t)

            h = rnn(x_t, h)
            x_t = F.dropout(h,
                            p=self.dropout,
                            training=self.training)
            x_t = self.lin(x_t)
            out_mels.append(x_t.unsqueeze(1))
            mel = x_t

        x = torch.cat(out_mels, dim=1)
        x = x.transpose(1, 2)
        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x, x_post, dur = x.squeeze(), x_post.squeeze(), dur.squeeze()
        x = x.cpu().data.numpy()
        x_post = x_post.cpu().data.numpy()
        dur = dur.cpu().data.numpy()

        return x, x_post, dur

    def pad(self, x, max_len):
        x = x[:, :max_len, :]
        x = F.pad(x, [0, 0, 0, max_len - x.size(1), 0, 0], 'constant', 0.0)
        return x

    def get_step(self):
        return self.step.data.item()

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell