import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

from models.tacotron import Tacotron
from trainer.common import Averager, TTSSession
from utils import hparams as hp
from utils.attention_score import attention_score
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_attention, plot_cos_matrix
from utils.dsp import reconstruct_waveform, np_now
from utils.files import unpickle_binary
from utils.paths import Paths


class TacoTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.tts_log, comment='v1')

    def train(self, model: Tacotron, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.tts_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=r, model_type='tacotron')
                session = TTSSession(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: Tacotron,
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        model.r = session.r
        simple_table([(f'Steps with r={session.r}', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr),
                      ('Outputs/Step (r)', model.r)])
        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (s_id, semb, x, m, ids, x_lens, mel_lens) in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                x, semb, m, s_id, x_lens, mel_lens = x.to(device), semb.to(device), m.to(device), s_id.to(device), x_lens.to(device), mel_lens.to(device)

                m1_hat, m2_hat, attention = model(x, m, semb)

                loc_score, sharp_score = attention_score(attention, mel_lens, r=model.r)
                att_score = torch.mean(loc_score * sharp_score)

                m1_loss = F.l1_loss(m1_hat, m)
                m2_loss = F.l1_loss(m2_hat, m)
                loss = m1_loss + m2_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                optimizer.step()
                loss_avg.add(loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.tts_checkpoint_every == 0:
                    ckpt_name = f'taco_step{k}K'
                    save_checkpoint('tts', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                if step % hp.tts_plot_every == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Attention_Score/train', att_score, model.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_loss, val_att_score = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Loss/val', val_loss, model.get_step())
            self.writer.add_scalar('AttentionScore/val', val_att_score, model.get_step())
            save_checkpoint('tts', self.paths, model, optimizer, is_silent=True)

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: Tacotron, val_set: Dataset) -> Tuple[float, float]:
        model.eval()
        val_loss = 0
        att_score_sum = 0
        device = next(model.parameters()).device
        for i, (s_id, semb, x, m, ids, x_lens, mel_lens) in enumerate(val_set, 1):
            x, semb, m, s_id, x_lens, mel_lens = x.to(device), semb.to(device), m.to(device), s_id.to(device), x_lens.to(device), mel_lens.to(device)

            with torch.no_grad():
                m1_hat, m2_hat, attention = model(x, m, semb)
                m1_loss = F.l1_loss(m1_hat, m)
                m2_loss = F.l1_loss(m2_hat, m)
                val_loss += m1_loss.item() + m2_loss.item()
            loc_score, sharp_score = attention_score(attention, mel_lens, r=model.r)
            att_score = torch.mean(loc_score * sharp_score)
            att_score_sum += att_score

        return val_loss / len(val_set), att_score_sum / len(val_set)

    @ignore_exception
    def generate_plots(self, model: Tacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        s_id, semb, x, m, ids, x_lens, m_lens = session.val_sample
        x, semb, m, s_id = x.to(device), semb.to(device), m.to(device), s_id.to(device)

        # plot speaker cosine similarity matrix
        speaker_emb_dict = unpickle_binary(self.paths.data / 'speaker_emb_dict.pkl')
        speaker_token_dict = unpickle_binary(self.paths.data / 'speaker_token_dict.pkl')
        token_speaker_dict = {v: k for k, v in speaker_token_dict.items()}
        speaker_ids = sorted(list(speaker_emb_dict.keys()))[:20]
        embeddings = [speaker_emb_dict[s_id] for s_id in speaker_ids]
        cos_mat = cosine_similarity(embeddings)
        np.fill_diagonal(cos_mat, 0)
        cos_mat_fig = plot_cos_matrix(cos_mat, labels=speaker_ids)
        self.writer.add_figure('Embedding_Metrics/speaker_cosine_dist', cos_mat_fig, model.step)

        for idx in range(len(hp.val_speaker_ids)):
            x_len = x_lens[idx]
            m_len = m_lens[idx]
            m1_hat, m2_hat, att = model(x[idx:idx+1, :x_len], m[idx:idx+1, :, :m_len], semb[idx:idx+1])
            att_np = np_now(att)
            m1_hat_np = np_now(m1_hat)
            m2_hat_np = np_now(m2_hat)
            gen_sid = int(s_id[idx].cpu())
            gen_semb = semb[idx].cpu()
            target_sid = token_speaker_dict[gen_sid]
            att = att_np[0]
            m1_hat = m1_hat_np[0, :, :]
            m2_hat = m2_hat_np[0, :, :]
            m_target_np = np_now(m)
            m_target = m_target_np[idx, :, :m_len]

            att_fig = plot_attention(att)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)
            m_target_fig = plot_mel(m_target)

            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/attention', att_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/target', m_target_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/postnet', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)
            target_wav = reconstruct_waveform(m_target)

            self.writer.add_audio(
                tag=f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

            m1_hat, m2_hat, att = model.generate(x[idx].tolist(), gen_semb, steps=m_lens[idx] + 20)
            att_fig = plot_attention(att)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)

            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/attention', att_fig, model.step)
            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/target', m_target_fig, model.step)
            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/postnet', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag=f'Generated_{idx}_SID_{target_sid}/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Generated_{idx}_SID_{target_sid}/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

