import random
import time
from typing import Tuple, Dict, Any

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.forward_tacotron import ForwardTacotron
from models.generator import Generator
from models.multiscale import MultiScaleDiscriminator
from trainer.common import Averager, TTSSession, MaskedL1, to_device, np_now
from utils.checkpoints import  save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_pitch
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.paths import Paths


class ForwardTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        self.train_cfg = config['forward_tacotron']['training']
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

        checkpoint = torch.load('checkpoints/melgan_asvoice2.pt', map_location=torch.device('cpu'))
        generator = Generator(80)
        generator.load_state_dict(checkpoint['model_g'])
        generator.train()
        disc = MultiScaleDiscriminator()
        disc.load_state_dict(checkpoint['model_d'])
        disc.train()
        self.generator = generator
        self.disc = disc

    def train(self, model: ForwardTacotron, optimizer: Optimizer) -> None:
        forward_schedule = self.train_cfg['schedule']
        forward_schedule = parse_schedule(forward_schedule)
        for i, session_params in enumerate(forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward',
                    max_mel_len=self.train_cfg['max_mel_len'],
                    filter_attention=self.train_cfg['filter_attention'],
                    filter_min_alignment=self.train_cfg['min_attention_alignment'],
                    filter_min_sharpness=self.train_cfg['min_attention_sharpness'])
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: ForwardTacotron,
                      optimizer: Optimizer, session: TTSSession) -> None:

        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        pitch_loss_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        self.generator.to(device)
        self.disc.to(device)

        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()

                pitch_zoneout_mask = torch.rand(batch['x'].size()) > self.train_cfg['pitch_zoneout']
                energy_zoneout_mask = torch.rand(batch['x'].size()) > self.train_cfg['energy_zoneout']

                pitch_target = batch['pitch'].detach().clone()
                energy_target = batch['energy'].detach().clone()
                batch['pitch'] = batch['pitch'] * pitch_zoneout_mask.to(device).float()
                batch['energy'] = batch['energy'] * energy_zoneout_mask.to(device).float()

                pred = model(batch)
                mel_len = pred['mel_post'].size(2)

                if mel_len < 60:
                    continue

                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])

                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], pitch_target.unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], energy_target.unsqueeze(1), batch['x_len'])


                loss = self.train_cfg['mel_loss_factor'] * (m1_loss + m2_loss) \
                       + self.train_cfg['dur_loss_factor'] * dur_loss \
                       + self.train_cfg['pitch_loss_factor'] * pitch_loss \
                       + self.train_cfg['energy_loss_factor'] * energy_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()

                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                pitch_loss_avg.add(pitch_loss.item())

                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | Pitch Loss: {pitch_loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.forward_checkpoints / f'forward_step{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                num_iter = self.train_cfg['gen_iter']
                loss_g_avg = 0
                for i in range(num_iter):
                    optimizer.zero_grad()
                    pred = model(batch)
                    mel_len = pred['mel_post'].size(2)
                    self.generator.zero_grad()
                    self.disc.zero_grad()
                    loss_g = 0
                    pred_start = random.randrange(0, mel_len-60)
                    audio = self.generator(pred['mel_post'][:, :, pred_start:pred_start+60])
                    disc_fake = self.disc(audio)
                    for feats_fake, score_fake in disc_fake:
                        loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    loss_g /= num_iter
                    loss_g.backward()
                    loss_g_avg += loss_g.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.train_cfg['clip_grad_norm'])
                    optimizer.step()

                self.writer.add_scalar('Gen_Loss/train', loss_g_avg, model.get_step())
                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model.get_step())
                self.writer.add_scalar('Pitch_Loss/train', pitch_loss, model.get_step())
                self.writer.add_scalar('Energy_Loss/train', energy_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_out = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', val_out['mel_loss'], model.get_step())
            self.writer.add_scalar('Duration_Loss/val', val_out['dur_loss'], model.get_step())
            self.writer.add_scalar('Pitch_Loss/val', val_out['pitch_loss'], model.get_step())
            self.writer.add_scalar('Energy_Loss/val', val_out['energy_loss'], model.get_step())
            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.forward_checkpoints / 'latest_model.pt')

            m_loss_avg.reset()
            duration_avg.reset()
            pitch_loss_avg.reset()
            print(' ')

    def evaluate(self, model: ForwardTacotron, val_set: DataLoader) -> Dict[str, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        pitch_val_loss = 0
        energy_val_loss = 0
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                pred = model(batch)
                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])
                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], batch['pitch'].unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], batch['energy'].unsqueeze(1), batch['x_len'])
                pitch_val_loss += pitch_loss
                energy_val_loss += energy_loss
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        return {
            'mel_loss': m_val_loss / len(val_set),
            'dur_loss': dur_val_loss / len(val_set),
            'pitch_loss': pitch_val_loss / len(val_set),
            'energy_loss': energy_val_loss / len(val_set)
        }

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)

        pred = model(batch)
        m1_hat = np_now(pred['mel'])[0, :600, :]
        m2_hat = np_now(pred['mel_post'])[0, :600, :]
        m_target = np_now(batch['mel'])[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_target_fig = plot_mel(m_target)
        pitch_fig = plot_pitch(np_now(batch['pitch'][0]))
        pitch_gta_fig = plot_pitch(np_now(pred['pitch'].squeeze()[0]))
        energy_fig = plot_pitch(np_now(batch['energy'][0]))
        energy_gta_fig = plot_pitch(np_now(pred['energy'].squeeze()[0]))

        self.writer.add_figure('Pitch/target', pitch_fig, model.step)
        self.writer.add_figure('Pitch/ground_truth_aligned', pitch_gta_fig, model.step)
        self.writer.add_figure('Energy/target', energy_fig, model.step)
        self.writer.add_figure('Energy/ground_truth_aligned', energy_gta_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/target', m_target_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)

        m2_hat_wav = self.dsp.griffinlim(m2_hat)
        target_wav = self.dsp.griffinlim(m_target)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        gen = model.generate(batch['x'][0:1, :batch['x_len'][0]])

        m1_hat_fig = plot_mel(gen['mel'])
        m2_hat_fig = plot_mel(gen['mel_post'])

        pitch_gen_fig = plot_pitch(np_now(gen['pitch'].squeeze()))
        energy_gen_fig = plot_pitch(np_now(gen['energy'].squeeze()))

        self.writer.add_figure('Pitch/generated', pitch_gen_fig, model.step)
        self.writer.add_figure('Energy/generated', energy_gen_fig, model.step)
        self.writer.add_figure('Generated/target', m_target_fig, model.step)
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.step)

        m2_hat_wav = self.dsp.griffinlim(m2_hat)

        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        with torch.no_grad():
            m2_hat_gen = self.generator(torch.tensor(m2_hat).to(device).unsqueeze(0))
        m2_hat_gen = m2_hat_gen.cpu().numpy()
        self.writer.add_audio(
            tag='Generated_Melgan/postnet_wav', snd_tensor=m2_hat_gen,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
