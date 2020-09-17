import time

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.text import clean_text, text_to_sequence

from models.tacotron import Tacotron
from trainer.common import Averager, TTSSession
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_attention, simple_plot
from utils.dsp import reconstruct_waveform, np_now
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
            for i, (x, m, ids, _) in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                x, m = x.to(device), m.to(device)
                
                m1_hat, m2_hat, attention, style_attn_scores = model(x=x, m=m, ref_mels=m)
                
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
                self.writer.add_scalar('Params/reduction_factor', session.r, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())
                
                stream(msg)
            
            val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Loss/val', val_loss, model.get_step())
            save_checkpoint('tts', self.paths, model, optimizer, is_silent=True)
            
            loss_avg.reset()
            duration_avg.reset()
            print(' ')
    
    def evaluate(self, model: Tacotron, val_set: Dataset) -> float:
        model.eval()
        val_loss = 0
        device = next(model.parameters()).device
        for i, (x, m, ids, _) in enumerate(val_set, 1):
            x, m = x.to(device), m.to(device)
            with torch.no_grad():
                m1_hat, m2_hat, attention, style_attn_scores = model(x, m, m)
                m1_loss = F.l1_loss(m1_hat, m)
                m2_loss = F.l1_loss(m2_hat, m)
                val_loss += m1_loss.item() + m2_loss.item()
        return val_loss / len(val_set)
    
    @ignore_exception
    def generate_plots(self, model: Tacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        cosim = torch.nn.CosineSimilarity(dim=0).to(device)
        sim_matrix = torch.zeros(model.gst.stl.embed.size(0), model.gst.stl.embed.size(0)).to(device)
        for i in range(model.gst.stl.embed.size(0)):
            for j in range(model.gst.stl.embed.size(0)):
                sim_matrix[i][j] = cosim(model.gst.stl.embed[i], model.gst.stl.embed[j])
        
        token_similarity_fig = simple_plot(np_now(sim_matrix))
        self.writer.add_figure(f'StyleTokenSimilarity', token_similarity_fig, model.step)
        
        generate_text = ['United Airlines five six three from Los Angeles to New Orleans has Landed.',
                         'Is that Utah travel agency?']
        generate_text = [text_to_sequence(clean_text(t)) for t in generate_text]
        mel_paths = ['/home/sysgen/datasets_unprocessed/LJSpeech-1.1/forward_taco_data/ref_mel.npy',
                     '/home/sysgen/datasets_unprocessed/LJSpeech-1.1/forward_taco_data/ref_mel_1.npy']
        for j, mel_path in enumerate(mel_paths):
            x, m, ids, lens = session.val_sample
            x, m = x.to(device), m.to(device)
            m_copy = m.detach().clone().to(device)
            ref_mel = np.load(mel_path)
            ref_mel = torch.from_numpy(ref_mel).to(device)
            batch_size = x.size(0)
            ref_mel_batch = ref_mel.repeat(batch_size, 1, 1)
            
            m1_hat, m2_hat, att, style_attn_scores = model(x, m, ref_mel_batch)
            att = np_now(att)[0]
            m1_hat = np_now(m1_hat)[0, :600, :]
            m2_hat = np_now(m2_hat)[0, :600, :]
            m = np_now(m)[0, :600, :]
            ref_mel_batch = np_now(ref_mel_batch)[0, :600, :]

            att_fig = plot_attention(att)
            style_attn_scores = style_attn_scores[:, 0, :, :]  # ([N_heads, 1, N_tokens])
            all_heads = torch.stack([t for t in style_attn_scores], dim=1)  # ([N_heads, N_tokens])
            all_heads = np_now(all_heads)
            style_attn_fig = plot_attention(all_heads)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)
            m_fig = plot_mel(m)
            ref_mel_fig = plot_mel(ref_mel_batch)

            self.writer.add_figure(f'Ground_Truth_Aligned/attention_{j}', att_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned/style_attention_{j}', style_attn_fig, model.step)
            self.writer.add_figure('Ground_Truth_Aligned/target', m_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned/reference_{j}', ref_mel_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned/linear_{j}', m1_hat_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned/postnet_{j}', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)
            target_wav = reconstruct_waveform(m)
            reference_wav = reconstruct_waveform(ref_mel_batch)

            self.writer.add_audio(
                tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'ReferenceWavs/{j}', snd_tensor=reference_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Ground_Truth_Aligned/postnet_wav_{j}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            ref_mel = ref_mel.repeat(1, 1, 1)
            m1_hat, m2_hat, att, style_attn_scores = model.generate(x[0].tolist(), steps=lens[0] + 20, ref_mel=ref_mel)
            style_attn_scores = style_attn_scores[:, 0, :, :]  # ([N_heads, 1, N_tokens])
            all_heads = torch.stack([t for t in style_attn_scores], dim=1)  # ([N_heads, N_tokens])
            all_heads = np_now(all_heads)
            style_attn_fig = plot_attention(all_heads)

            att_fig = plot_attention(att)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)

            self.writer.add_figure(f'Generated/attention_{j}', att_fig, model.step)
            self.writer.add_figure(f'Generated/style_attention_{j}', style_attn_fig, model.step)
            self.writer.add_figure('Generated/target', m_fig, model.step)
            self.writer.add_figure(f'Generated/postnet_{j}', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag='Generated/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Generated/postnet_wav_{j}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

            m1_hat, m2_hat, att,style_attn_scores= model.generate(generate_text[j], steps=ref_mel.size(2) + 20, ref_mel=ref_mel)
            style_attn_scores = style_attn_scores[:, 0, :, :]  # ([N_heads, 1, N_tokens])
            all_heads = torch.stack([t for t in style_attn_scores], dim=1)  # ([N_heads, N_tokens])
            all_heads = np_now(all_heads)
            style_attn_fig = plot_attention(all_heads)
            att_fig = plot_attention(att)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)

            self.writer.add_figure(f'GeneratedTest/attention_{j}', att_fig, model.step)
            self.writer.add_figure(f'GeneratedTest/style_attention_{j}', style_attn_fig, model.step)
            self.writer.add_figure(f'GeneratedTest/postnet_{j}', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag=f'GeneratedTest/postnet_wav_{j}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            m1_hat, m2_hat, att, style_attn_scores = model.generate(generate_text[j], steps=ref_mel.size(2) + 20,
                                                                    ref_mel=m_copy[0:1])
            style_attn_scores = style_attn_scores[:, 0, :, :]  # ([N_heads, 1, N_tokens])
            all_heads = torch.stack([t for t in style_attn_scores], dim=1) # ([N_heads, N_tokens])
            all_heads = np_now(all_heads)
            style_attn_fig = plot_attention(all_heads)
            att_fig = plot_attention(att)
            m2_hat_fig = plot_mel(m2_hat)

            self.writer.add_figure(f'GeneratedTestNoStyle/attention_{j}', att_fig, model.step)
            self.writer.add_figure(f'GeneratedTestNoStyle/style_attention_{j}', style_attn_fig, model.step)
            self.writer.add_figure(f'GeneratedTestNoStyle/postnet_{j}', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag=f'GeneratedTestNoStyle/postnet_wav_{j}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

