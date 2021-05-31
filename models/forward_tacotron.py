from pathlib import Path
from typing import Union, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.text.symbols import phonemes

MEL_PAD_VALUE = -11.5129


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        return self.expand(x, dur)

    @staticmethod
    def build_index(duration: torch.tensor, x: torch.tensor) -> torch.tensor:
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

    def expand(self, x: torch.tensor, dur: torch.tensor) -> torch.tensor:
        idx = self.build_index(dur, x)
        y = torch.gather(x, 1, idx)
        return y


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.rnn = nn.GRU(emb_dim, rnn_dims, batch_first=True, bidirectional=True)
        self.rnn_2 = nn.GRU(2 * rnn_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.tensor,
                x_lens: torch.tensor = None,
                alpha=1.0) -> torch.tensor:
        x = self.embedding(x)
        if x_lens is not None:
            x = pack_padded_sequence(x, lengths=x_lens, batch_first=True,
                                     enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = self.rnn_2(x)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, padding_value=0.0, batch_first=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x / alpha


class ConvResNet(nn.Module):

    def __init__(self, in_dims: int, layers=5, conv_dims=512) -> None:
        super().__init__()
        self.first_conv = BatchNormConv(in_dims, conv_dims, 5, activation=torch.tanh)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.tanh) for _ in range(layers - 1)
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.transpose(1, 2)
        x = self.first_conv(x)
        for conv in self.convs:
            x_res = x
            x = conv(x)
            x = x_res + x
        x = x.transpose(1, 2)
        return x


class ConvLstm(nn.Module):

    def __init__(self, in_dims: int, layers=3, conv_dims=512, lstm_dims=512) -> None:
        super().__init__()
        self.first_conv = BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu)
        self.last_conv = BatchNormConv(conv_dims, conv_dims, 5, activation=None)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu) for _ in range(layers - 2)
        ])
        self.lstm = nn.LSTM(conv_dims, lstm_dims, batch_first=True, bidirectional=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.transpose(1, 2)
        x = self.first_conv(x)
        for conv in self.convs:
            x = conv(x)
        x = self.last_conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return x


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 series_embed_dims: int,
                 num_chars: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_emb_dims: int,
                 pitch_proj_dropout: float,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_emb_dims: int,
                 energy_proj_dropout: float,
                 prenet_conv_dims: int,
                 prenet_lstm_dims: int,
                 main_conv_dims: int,
                 main_lstm_dims: int,
                 postnet_conv_dims: int,
                 dropout: float,
                 n_mels: int):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          emb_dim=series_embed_dims,
                                          rnn_dims=pitch_rnn_dims,
                                          dropout=pitch_dropout)
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           emb_dim=series_embed_dims,
                                           rnn_dims=energy_rnn_dims,
                                           dropout=energy_dropout)
        self.prenet = ConvLstm(in_dims=embed_dims,
                               conv_dims=prenet_conv_dims,
                               lstm_dims=prenet_lstm_dims)
        self.main_net = ConvLstm(in_dims=2 * prenet_lstm_dims + pitch_emb_dims + energy_emb_dims,
                                 conv_dims=main_conv_dims,
                                 lstm_dims=main_lstm_dims)
        self.lin = torch.nn.Linear(2 * main_lstm_dims, n_mels)
        self.postnet = ConvResNet(in_dims=n_mels, conv_dims=postnet_conv_dims)
        self.dropout = dropout
        self.post_proj = nn.Linear(postnet_conv_dims, n_mels, bias=False)
        self.pitch_emb_dims = pitch_emb_dims
        self.energy_emb_dims = energy_emb_dims
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

        if pitch_emb_dims > 0:
            self.pitch_proj = nn.Sequential(
                nn.Conv1d(1, pitch_emb_dims, kernel_size=3, padding=1),
                nn.Dropout(pitch_proj_dropout))
        if energy_emb_dims > 0:
            self.energy_proj = nn.Sequential(
                nn.Conv1d(1, energy_emb_dims, kernel_size=3, padding=1),
                nn.Dropout(energy_proj_dropout))

    def forward(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        x_lens = batch['x_len'].cpu()
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        dur_hat = self.dur_pred(x, x_lens=x_lens).squeeze(-1)
        pitch_hat = self.pitch_pred(x, x_lens=x_lens).transpose(1, 2)
        energy_hat = self.energy_pred(x, x_lens=x_lens).transpose(1, 2)

        x = self.embedding(x)
        x = self.prenet(x)

        if self.pitch_emb_dims > 0:
            pitch_proj = self.pitch_proj(pitch)
            pitch_proj = pitch_proj.transpose(1, 2)
            x = torch.cat([x, pitch_proj], dim=-1)

        if self.energy_emb_dims > 0:
            energy_proj = self.energy_proj(energy)
            energy_proj = energy_proj.transpose(1, 2)
            x = torch.cat([x, energy_proj], dim=-1)

        x = self.lr(x, dur)
        x = self.main_net(x)
        x = self.lin(x)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)

        x = x.transpose(1, 2)
        x_post = x_post.transpose(1, 2)
        x_post = self.pad(x_post, mel.size(2))
        x = self.pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 energy_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,

                 ) -> Dict[str, np.array]:
        self.eval()

        dur = self.dur_pred(x, alpha=alpha)
        dur = dur.squeeze(2)

        # Fixing breaking synth of silent texts
        if torch.sum(dur) <= 0:
            dur = torch.full(x.size(), fill_value=2, device=x.device)

        pitch_hat = self.pitch_pred(x).transpose(1, 2)
        pitch_hat = pitch_function(pitch_hat)

        energy_hat = self.energy_pred(x).transpose(1, 2)
        energy_hat = energy_function(energy_hat)

        x = self.embedding(x)
        x = self.prenet(x)

        if self.pitch_emb_dims > 0:
            pitch_hat_proj = self.pitch_proj(pitch_hat).transpose(1, 2)
            x = torch.cat([x, pitch_hat_proj], dim=-1)

        if self.energy_emb_dims > 0:
            energy_hat_proj = self.energy_proj(energy_hat).transpose(1, 2)
            x = torch.cat([x, energy_hat_proj], dim=-1)

        x = self.lr(x, dur)
        x = self.main_net(x)
        x = self.lin(x)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)

        x = x.transpose(1, 2)
        x_post = x_post.transpose(1, 2)

        x, x_post, dur = x.squeeze(), x_post.squeeze(), dur.squeeze()
        x = x.cpu().data.numpy()
        x_post = x_post.cpu().data.numpy()
        dur = dur.cpu().data.numpy()

        return {'mel': x, 'mel_post': x_post, 'dur': dur,
                'pitch': pitch_hat, 'energy': energy_hat}

    def pad(self, x: torch.tensor, max_len: int) -> torch.tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', MEL_PAD_VALUE)
        return x

    def get_step(self) -> int:
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
        model_config = config['forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return ForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = ForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model