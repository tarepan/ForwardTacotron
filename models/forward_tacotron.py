from pathlib import Path
from typing import Union
import random
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d

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


class DurationPredictor(nn.Module):

    def __init__(self, in_dims, conv_dims=256):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            nn.BatchNorm1d(in_dims),
            nn.ReLU(),
            nn.Conv1d(in_dims, conv_dims, kernel_size=1),
            nn.BatchNorm1d(conv_dims),
            nn.ReLU(),
            nn.Conv1d(conv_dims, 1, kernel_size=1),
            nn.ReLU(),
        ])

    def forward(self, x, alpha=1.0):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)

        return x


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=True)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ConvStack(nn.Module):

    def __init__(self, channel, layers=10):
        super(ConvStack, self).__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(channel) for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, channel):
        super(ConvBlock, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.Conv1d(channel, channel, kernel_size=3, dilation=a, padding=a),
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.Conv1d(channel, channel, kernel_size=3, dilation=b, padding=b),
            )
            for a, b in [(1, 2), (4, 8), (16, 32)]
        ])

    def forward(self, x):
        for block in self.blocks:
            block_inputs = x
            x = block(x)
            x += block_inputs
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
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(256, conv_dims=durpred_conv_dims)
        self.prenet = torch.nn.Conv1d(embed_dims, 256, 7, padding=3)

        self.lin = torch.nn.Linear(2 * rnn_dim, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = torch.nn.Conv1d(256, 256, 7, padding=3)
        self.post_proj = nn.Linear(256, n_mels, bias=False)

    def forward(self, x, mel, x_lens, mel_lens, durs, out_offset=0, out_seq_len=None):
        device = next(self.parameters()).device

        if self.training:
            self.step += 1

        x = self.embedding(x)
        x = x.transpose(1, 2)
        x  = self.prenet(x)
        x = x.transpose(1, 2)
        x_p = x

        token_lengths = self.dur_pred(x)
        token_lengths = token_lengths.squeeze(1)

        sequence_length = x.shape[1]
        mask = torch.arange(sequence_length)[None, :].to(device) < x_lens[:, None]
        mask = mask.to(device)

        token_ends = torch.cumsum(token_lengths, dim=1)
        aligned_lengths = token_ends.gather(1, x_lens.unsqueeze(1)-1).squeeze()

        token_centres = token_ends - (token_lengths / 2.)

        out_seq_len = mel.shape[-1]
        out_pos = torch.arange(0, out_seq_len)[None, :] + out_offset

        out_pos = out_pos.to(device)
        out_pos = out_pos[:, :, None].float()
        diff = token_centres[:, None, :] - out_pos
        logits = - (diff ** 2 / 10.)
        logits_inv_mask = 1. - mask[:, None, :].float()

        masked_logits = logits - 1e9 * logits_inv_mask
        weights = torch.softmax(masked_logits, dim=2)

        x = torch.einsum('bij,bjk->bik', weights, x_p)
        x = x.transpose(1, 2)
        x_post = self.postnet(x)
        x_post = x_post.transpose(1, 2)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self.pad(x_post, mel.size(2))

        return x_post, x_post, aligned_lengths, token_lengths

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
        x, _ = self.lstm(x)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
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
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', -11.51)
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

