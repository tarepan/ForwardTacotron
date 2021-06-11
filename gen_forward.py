import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils.display import simple_table
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


def load_forward_taco(checkpoint_path: str) -> Tuple[ForwardTacotron, Dict[str, Any]]:
    print(f'Loading tts checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    tts_model = ForwardTacotron.from_config(config)
    tts_model.load_state_dict(checkpoint['model'])
    print(f'Loaded forward taco with step {tts_model.get_step()}')
    return tts_model, config


def load_wavernn(checkpoint_path: str) -> Tuple[WaveRNN, Dict[str, Any]]:
    print(f'Loading voc checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    voc_model = WaveRNN.from_config(config)
    voc_model.load_state_dict(checkpoint['model'])
    print(f'Loaded model with step {voc_model.get_step()}')
    return voc_model, config


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', default=None, type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--checkpoint', type=str, default=None, help='[string/path] path to .pt model file.')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams. Only'
                                                                                'used if no checkpoint is set.')
    parser.add_argument('--alpha', type=float, default=1., help='Parameter for controlling length regulator for speedup '
                                                                'or slow-down of generated speech, e.g. alpha=2.0 is double-time')
    parser.add_argument('--amp', type=float, default=1., help='Parameter for controlling pitch amplification')

    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(dest='vocoder')
    wr_parser = subparsers.add_parser('wavernn')
    wr_parser.add_argument('--overlap', '-o', default=550,  type=int, help='[int] number of crossover samples')
    wr_parser.add_argument('--target', '-t', default=11_000, type=int, help='[int] number of samples in each batch index')
    wr_parser.add_argument('--voc_checkpoint', type=str, help='[string/path] Load in different WaveRNN weights')

    gl_parser = subparsers.add_parser('griffinlim')
    mg_parser = subparsers.add_parser('melgan')

    args = parser.parse_args()

    assert args.vocoder in {'griffinlim', 'wavernn', 'melgan'}, \
        'Please provide a valid vocoder! Choices: [\'griffinlim\', \'wavernn\', \'melgan\']'

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        config = read_config(args.config)
        paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
        checkpoint_path = paths.forward_checkpoints / 'latest_model.pt'

    tts_model, config = load_forward_taco(checkpoint_path)
    dsp = DSP.from_config(config)

    voc_model, voc_dsp = None, None
    if args.vocoder == 'wavernn':
        voc_model, voc_config = load_wavernn(args.voc_checkpoint)
        voc_dsp = DSP.from_config(voc_config)

    out_path = Path('model_outputs')
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tts_model.to(device)
    cleaner = Cleaner.from_config(config)
    tokenizer = Tokenizer()

    print(f'Using device: {device}\n')
    if args.input_text:
        texts = [args.input_text]
    else:
        with open('sentences.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()

    tts_k = tts_model.get_step() // 1000

    simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                  ('Vocoder Type', args.vocoder)])
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

    def __init__(self, num_chars, emb_dim=64, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.tensor,
                x_lens: torch.tensor = None,
                alpha=1.0) -> torch.tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        if x_lens is not None:
            x = pack_padded_sequence(x, lengths=x_lens, batch_first=True,
                                     enforce_sorted=False)
        x, _ = self.rnn(x)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, padding_value=0.0, batch_first=True)
        x = self.lin(x)
        return x / alpha


class ConvGru(nn.Module):

    def __init__(self, in_dims: int, layers=3, conv_dims=512, gru_dims=512) -> None:
        super().__init__()
        self.first_conv = BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu)
        self.last_conv = BatchNormConv(conv_dims, conv_dims, 5, activation=None)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu) for _ in range(layers - 2)
        ])
        self.lstm = nn.GRU(conv_dims, gru_dims, batch_first=True, bidirectional=True)

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
                 durpred_conv_dims: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_conv_dims: int,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_emb_dims: int,
                 pitch_proj_dropout: float,
                 energy_conv_dims: int,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_emb_dims: int,
                 energy_proj_dropout: float,
                 prenet_gru_dims: int,
                 main_conv_dims: int,
                 main_gru_dims: int,
                 postnet_conv_dims: int,
                 postnet_gru_dims: int,
                 dropout: float,
                 n_mels: int):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        conv_dims=durpred_conv_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)
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
        self.prenet = nn.GRU(embed_dims, prenet_gru_dims, batch_first=True, bidirectional=True)
        self.main_net = ConvGru(in_dims=2 * prenet_gru_dims + pitch_emb_dims + energy_emb_dims,
                                conv_dims=main_conv_dims,
                                gru_dims=main_gru_dims)
        self.lin = torch.nn.Linear(2 * main_gru_dims, n_mels)
        self.postnet = ConvGru(in_dims=n_mels,
                               conv_dims=postnet_conv_dims,
                               gru_dims=postnet_gru_dims)
        self.dropout = dropout
        self.post_proj = nn.Linear(2*postnet_gru_dims, n_mels, bias=False)
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
        x, _ = self.prenet(x)

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
        x_post = x + x_post

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
        x, _ = self.prenet(x)

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
        x_post = x + x_post

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

    # simple amplification of pitch
    pitch_function = lambda x: x * args.amp
    energy_function = lambda x: x

    for i, x in enumerate(texts, 1):
        print(f'\n| Generating {i}/{len(texts)}')
        text = x
        x = cleaner(x)
        x = tokenizer(x)
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        wav_name = f'{i}_forward_{tts_k}k_alpha{args.alpha}_amp{args.amp}_{args.vocoder}_inf'

        gen = tts_model.generate(x=x,
                                 alpha=args.alpha,
                                 pitch_function=pitch_function,
                                 energy_function=energy_function)

        m = gen['mel_post']
        if args.vocoder == 'melgan':
            m = torch.tensor(m).unsqueeze(0)
            torch.save(m, out_path / f'{wav_name}.mel')
        if args.vocoder == 'wavernn':
            m = torch.tensor(m).unsqueeze(0)
            wav = voc_model.generate(mels=m,
                                     batched=True,
                                     target=args.target,
                                     overlap=args.overlap,
                                     mu_law=voc_dsp.mu_law)
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')
        elif args.vocoder == 'griffinlim':
            wav = dsp.griffinlim(m)
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')

    print('\n\nDone.\n')
