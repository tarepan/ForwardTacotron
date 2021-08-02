from pathlib import Path
from typing import Union, Callable, Dict, Any, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, TransformerEncoder, MultiheadAttention, Sequential
from torch.nn.utils.rnn import pad_sequence

from models.common_layers import CBHG
from utils.text.symbols import phonemes


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        x_expanded = []
        for i in range(x.size(0)):
            x_exp = torch.repeat_interleave(x[i], (dur[i] + 0.5).long(), dim=0)
            x_expanded.append(x_exp)
        x_expanded = pad_sequence(x_expanded, padding_value=0., batch_first=True)
        return x_expanded


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def make_len_mask(inp: torch.Tensor) -> torch.Tensor:
    return (inp == 0).transpose(0, 1)


class TransformerEncoderConvLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderConvLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, dim_feedforward, 9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(dim_feedforward, d_model, 1, stride=1, padding=0)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderConvLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src).transpose(0, 1).transpose(1, 2)
        src2 = self.conv1(src)
        src2 = self.activation(src2)
        src2 = self.conv2(src2)
        src = src + self.dropout2(src2)
        src = src.transpose(1, 2).transpose(0, 1)
        src = self.norm2(src)
        return src


class ForwardTransformer(torch.nn.Module):

    def __init__(self,
                 d_model=256,
                 d_fft=512,
                 layers=6,
                 dropout=0.1,
                 heads=4) -> None:
        super(ForwardTransformer, self).__init__()

        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderConvLayer(d_model=d_model,
                                                    nhead=heads,
                                                    dim_feedforward=d_fft,
                                                    dropout=dropout)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

    def forward(self, x: torch.Tensor, src_pad_mask=None) -> torch.Tensor:         # shape: [N, T]

        x = x.transpose(0, 1)        # shape: [T, N]
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = x.transpose(0, 1)
        return x


class SeriesPredictor(nn.Module):

    def __init__(self,
                 d_model: int,
                 conv_dim: int,
                 dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, conv_dim, kernel_size=3, padding=1)
        self.lnorm1 = nn.LayerNorm(conv_dim)
        self.conv2 = nn.Conv1d(d_model, conv_dim, kernel_size=3, padding=1)
        self.lnorm2 = nn.LayerNorm(conv_dim)
        self.lin = nn.Linear(conv_dim, 1)

    def forward(self,
                x: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.lnorm1(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.lnorm2(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 num_chars: int,
                 series_dropout: float,
                 series_conv_dim: int,
                 pitch_strength: float,
                 energy_strength: float,
                 postnet_layers: int,
                 postnet_heads: int,
                 postnet_fft: int,
                 postnet_dropout: float,
                 prenet_layers: int,
                 prenet_heads: int,
                 prenet_fft: int,
                 d_model: int,
                 prenet_dropout: float,
                 n_mels: int,
                 padding_value=-11.5129):
        super().__init__()
        self.padding_value = padding_value
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(d_model,
                                        conv_dim=series_conv_dim,
                                        dropout=series_dropout)
        self.pitch_pred = SeriesPredictor(d_model,
                                          conv_dim=series_conv_dim,
                                          dropout=series_dropout)
        self.energy_pred = SeriesPredictor(d_model,
                                           conv_dim=series_conv_dim,
                                           dropout=series_dropout)

        self.embedding = Embedding(num_embeddings=num_chars, embedding_dim=d_model)

        self.prenet = ForwardTransformer(heads=prenet_heads, dropout=prenet_dropout,
                                         d_model=d_model, d_fft=prenet_fft, layers=prenet_layers)

        self.postnet = ForwardTransformer(heads=postnet_heads, dropout=postnet_dropout,
                                          d_model=d_model, d_fft=postnet_fft, layers=postnet_layers)

        self.lin = torch.nn.Linear(d_model, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength
        self.pitch_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        mel_lens = batch['mel_len']
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        len_mask = make_len_mask(x.transpose(0, 1))

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=len_mask)

        dur_hat = self.dur_pred(x).squeeze(-1)
        pitch_hat = self.pitch_pred(x).transpose(1, 2)
        energy_hat = self.energy_pred(x).transpose(1, 2)

        pitch_proj = self.pitch_proj(pitch)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur)

        len_mask = torch.zeros((x.size(0), x.size(1))).bool().to(x.device)
        for i, mel_len in enumerate(mel_lens):
            len_mask[i, mel_len:] = True

        x = self.postnet(x, src_pad_mask=len_mask)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.pad(x, mel.size(2))
        x = self.pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, np.array]:
        self.eval()

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=None)

        dur = self.dur_pred(x, alpha=alpha)
        dur = dur.squeeze(2)

        # Fixing breaking synth of silent texts
        if torch.sum(dur) <= 0:
            dur = torch.full(x.size(), fill_value=2, device=x.device)

        pitch_hat = self.pitch_pred(x).transpose(1, 2)
        pitch_hat = pitch_function(pitch_hat)

        energy_hat = self.energy_pred(x).transpose(1, 2)
        energy_hat = energy_function(energy_hat)

        pitch_proj = self.pitch_proj(pitch_hat)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy_hat)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur)

        x = self.postnet(x, src_pad_mask=None)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x, x_post, dur = x.squeeze(), x.squeeze(), dur.squeeze()
        x = x.cpu().data.numpy()
        x_post = x_post.cpu().data.numpy()
        dur = dur.cpu().data.numpy()

        return {'mel': x, 'mel_post': x_post, 'dur': dur,
                'pitch': pitch_hat, 'energy': energy_hat}

    def pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
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

if __name__ == '__main__':
    from dp.utils.io import read_config

    config = read_config('../config.yaml')
    tts_model = ForwardTacotron.from_config(config)
    tts_model.eval()

    model_script = torch.jit.script(tts_model)

    x = torch.ones((1, 5)).long()
    y = model_script.generate_jit(x)
    print(y)