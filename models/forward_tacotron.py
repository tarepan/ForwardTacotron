from pathlib import Path
from typing import Union, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.common_layers import CBHG, LengthRegulator
from utils.text.symbols import phonemes


class SeriesPredictor(nn.Module):
    """
    Encode a index series into a scalar series.

    embed - [Conv1d_s1-ReLU-BN-dropout]x3 - bidiGRU - segFC1
    """

    def __init__(self, num_chars, emb_dim=64, conv_dims=256, rnn_dims=64, dropout=0.5):
        """
        Args:
            num_chars: Size of discrete input codebook
            emb_dim: Embedding dimension
            conv_dims: Convolution's channel dimension
            rnn_dims: RNN hidden unit dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        # [Conv1d_k5s1-ReLU-BN]x3
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim,   conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        # 1-layer bidiGRU
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        # Segmental summary, vector2scalar
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):
    """Conv1d_s1-(ReLU)-BN"""
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
    """
    ForwardTacotron V2 (original ForwardTacotron + Energy Prediction + Pitch Prediction).
    
    Encode phoneme series, dynamically upsample, then generate mel-spectrogram.
    In contract to Transformer-based FastSpeech/FastPitch, this model is RNN-based.
    
    Model name comes from official demo.
        > NEW (14.05.2021): Forward Tacotron V2 (Energy + Pitch) + HiFiGAN Vocoder
        > [ref](https://as-ideas.github.io/ForwardTacotron/#new-14052021-forward-tacotron-v2-energy--pitch--hifigan-vocoder)
    
    phoneme_series::(B, T)
      => (CBHG Encoder)         => latent_series::(B, T, Z)
      => (d/p/e predictors)     => duration::(B, T, 1)  pitch::(B, T, 1)  energy::(B, T, 1)
      => (p/e proj + d LenReg)  => conditioning_series::(B, T_mel, Z)
      => (LSTM DecMain + segFC) => mel_spectrum_prototype_series::(B, T_mel, Freq_mel)
      => (CBHG DecPost)         => mel_spectrogram::(B, T_mel, Freq_mel)
    """

    def __init__(self,
                 embed_dims: int,           # Phoneme embedding dimension
                 series_embed_dims: int,    # Phoneme embedding dimension for duration/pitch/energy prediction
                 num_chars: int,            # Size of discrete phoneme series 
                 durpred_conv_dims: int,    # Convolution channel dimension of duration predictor
                 durpred_rnn_dims: int,     # RNN hidden dimension          of duration predictor
                 durpred_dropout: float,    # Dropout probability           of duration predictor
                 pitch_conv_dims: int,      # Convolution channel dimension of pitch predictor
                 pitch_rnn_dims: int,       # RNN hidden dimension          of pitch predictor
                 pitch_dropout: float,      # Dropout probability           of pitch predictor
                 pitch_strength: float,     # Scale factor                  of pitch prediction
                 energy_conv_dims: int,     # Convolution channel dimension of energy predictor
                 energy_rnn_dims: int,      # RNN hidden dimension          of energy predictor
                 energy_dropout: float,     # Dropout probability           of energy predictor
                 energy_strength: float,    # Scale factor                  of energy prediction
                 rnn_dims: int,             # Decoder MainNet RNN's hidden dimension
                 prenet_dims: int,          # Dimension of PreNet's default (Conv, GRU, output)
                 prenet_k: int,             # Maximum kernel size of PreNet-CBHG's multi-resolution ConvBank
                 postnet_num_highways: int, # Layer number of PreNet-CBHG's Highway Network
                 prenet_dropout: float,     # PreNet dropout probability
                 postnet_dims: int,         # Dimension of PostNet's default (Conv, GRU, output)
                 postnet_k: int,            # Maximum kernel size of PostNet-CBHG's multi-resolution ConvBank
                 prenet_num_highways: int,  # Layer number of PostNet-CBHG's Highway Network
                 postnet_dropout: float,    # PostNet dropout probability
                 n_mels: int,               # Number of mel frequency bins
                 padding_value=-11.5129):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.padding_value = padding_value
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

        # Encoder: phoneme_index::(T,) => latent_series::(T, Latent)
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
                           num_highways=prenet_num_highways,
                           dropout=prenet_dropout)

        # Pitch/Energy Predictor
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          emb_dim=series_embed_dims,
                                          conv_dims=pitch_conv_dims,
                                          rnn_dims=pitch_rnn_dims,
                                          dropout=pitch_dropout)
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           emb_dim=series_embed_dims,
                                           conv_dims=energy_conv_dims,
                                           rnn_dims=energy_rnn_dims,
                                           dropout=energy_dropout)
        # `2*` comes from CBHG's 'bidirectional' RNN output
        self.pitch_proj = nn.Conv1d(1, 2 * prenet_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 2 * prenet_dims, kernel_size=3, padding=1)
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength

        # Length Regulator: Dynamic Upsampling
        ## Regulator: Upsample a sequence with given duration 
        self.lr = LengthRegulator()
        ## Predictor: Predict phoneme segment duration
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        conv_dims=durpred_conv_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)

        # Decoder
        ## PreNet: removed
        pass
        ## MainNet
        ### 1-layer non-AR bidiLSTM
        self.lstm = nn.LSTM(2 * prenet_dims,
                            rnn_dims,
                            batch_first=True,
                            bidirectional=True)
        ### segFC projection: o_lstm => mel-spectrum prototype
        self.lin = torch.nn.Linear(2 * rnn_dims, n_mels)
        ## PostNet: mel-spectrogram prototype => mel-spectrogram
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=postnet_num_highways,
                            dropout=postnet_dropout)
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train the model with duration/pitch/energy teacher-forcing.
        """
        # Phoneme index series
        x = batch['x']
        mel = batch['mel']
        mel_lens = batch['mel_len']
        # Reference signals for teacher-forcing
        dur = batch['dur']
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        ### Just prediction for predictor training
        dur_hat = self.dur_pred(x).squeeze(-1)
        pitch_hat = self.pitch_pred(x).transpose(1, 2)
        energy_hat = self.energy_pred(x).transpose(1, 2)

        # Encoder
        ## Common
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.prenet(x)
        ## Pitch Prediction
        ### Pitch input as teacher forcing
        pitch_proj = self.pitch_proj(pitch)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength
        ## Energy Prediction
        ### Energy input as teacher forcing
        energy_proj = self.energy_proj(energy)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        # Length Regulator
        ## Duration input as teacher forcing
        x = self.lr(x, dur)
        x = pack_padded_sequence(x, lengths=mel_lens.cpu(), enforce_sorted=False,
                                 batch_first=True)

        # Decoder
        ## MainNet
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, padding_value=self.padding_value, batch_first=True)
        x = self.lin(x)
        x = x.transpose(1, 2)
        ## PostNet
        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self._pad(x_post, mel.size(2))
        x = self._pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # Duration Prediction
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)

            # Pitch Prediction
            pitch_hat = self.pitch_pred(x).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)

            # Energy Prediction
            energy_hat = self.energy_pred(x).transpose(1, 2)
            energy_hat = energy_function(energy_hat)

            # Encode/Upsample/Decode => mel-spec
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    @torch.jit.export
    def generate_jit(self,
                     x: torch.Tensor,
                     alpha: float = 1.0,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x).transpose(1, 2) * beta
            energy_hat = self.energy_pred(x).transpose(1, 2)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      energy_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          (B, T): Phoneme index series
            dur_hat    (B, T, 1): Predicted duration series
            pitch_hat  (B, T, 1): Predicted Pitch series
            energy_hat (B, T, 1): Predicted Energy series
        """

        # Encoder
        ## Common
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.prenet(x)
        ## Projected pitch sum
        pitch_proj = self.pitch_proj(pitch_hat)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength
        ## Projected energy sum
        energy_proj = self.energy_proj(energy_hat)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        # Length regulator
        x = self.lr(x, dur_hat)

        # Decoder
        ## MainNet
        x, _ = self.lstm(x)
        x = self.lin(x)
        x = x.transpose(1, 2)
        ## PostNet
        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return {'mel': x, 'mel_post': x_post, 'dur': dur_hat,
                'pitch': pitch_hat, 'energy': energy_hat}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x


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
