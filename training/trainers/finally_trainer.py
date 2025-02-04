import torch
from utils.data_utils import mel_spectrogram, debug_msg
from utils.model_utils import requires_grad, closest_power_of_two
from training.trainers.base_trainer import BaseTrainer

from training.trainers.base_trainer import gan_trainers_registry

class FinallyTrainer(BaseTrainer):
    def synthesize_wavs(self, batch):
        pass

@gan_trainers_registry.add_to_registry(name='finally_stage1_trainer')
class FinallyStage1Trainer(FinallyTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = "finally_gen"

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        gen_wav = gen(input_wav)
        gen_optimizer.zero_grad()

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_wav=gen_wav,
                real_wav=real_wav
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict}

@gan_trainers_registry.add_to_registry(name='finally_stage2_trainer')
class FinallyStage2Trainer(FinallyTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = "finally_gen"
        self.disc_name = "ms-stft_disc"
        self.mel_kwargs = dict(
            n_fft=config.mel.n_fft,
            num_mels=config.mel.num_mels,
            sampling_rate=config.mel.sampling_rate,
            hop_size=config.mel.hop_size,
            win_size=config.mel.win_size,
            fmin=config.mel.fmin,
            fmax=config.mel.fmax_for_loss
        )

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]
        disc = self.models[self.disc_name]
        disc_optimizer = self.optimizers[self.disc_name]
        disc_loss_builder = self.loss_builders[self.disc_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        gen_wav = gen(input_wav)
        gen_mel = mel_spectrogram(gen_wav.squeeze(1), **self.mel_kwargs)
        real_mel = mel_spectrogram(real_wav.squeeze(1), **self.mel_kwargs)

        requires_grad(disc, True)
        disc.zero_grad()

        ssd_real_out, ssd_gen_out, _, _ = disc(real_wav, gen_wav.detach())
        disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
            'ms-stft_disc': dict(
                discs_real_out=ssd_real_out,
                discs_gen_out=ssd_gen_out
            )
        }, tl_suffix='ssd')
        disc_loss.backward()
        disc_optimizer.step()

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        _, disc_gen_out, disc_fmaps_real, disc_fmaps_gen = disc(real_wav, gen_wav)

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_mel=gen_mel,
                real_mel=real_mel
            ),
            'ms-stft_disc': dict(
                discs_gen_out=ssd_gen_out,
                fmaps_real=disc_fmaps_real,
                fmaps_gen=disc_fmaps_gen
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict, **disc_losses_dict}

@gan_trainers_registry.add_to_registry(name='finally_stage3_trainer')
class FinallyStage3Trainer(FinallyTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = "finally_gen"
        self.disc_name = "ms-stft_disc"
        self.mel_kwargs = dict(
            n_fft=config.mel.n_fft,
            num_mels=config.mel.num_mels,
            sampling_rate=config.mel.sampling_rate,
            hop_size=config.mel.hop_size,
            win_size=config.mel.win_size,
            fmin=config.mel.fmin,
            fmax=config.mel.fmax_for_loss
        )

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]
        disc = self.models[self.disc_name]
        disc_optimizer = self.optimizers[self.disc_name]
        disc_loss_builder = self.loss_builders[self.disc_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        gen_wav = gen(input_wav)
        gen_mel = mel_spectrogram(gen_wav.squeeze(1), **self.mel_kwargs)
        real_mel = mel_spectrogram(real_wav.squeeze(1), **self.mel_kwargs)

        requires_grad(disc, True)
        disc.zero_grad()

        ssd_real_out, ssd_gen_out, _, _ = disc(real_wav, gen_wav.detach())
        disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
            'ms-stft_disc': dict(
                discs_real_out=ssd_real_out,
                discs_gen_out=ssd_gen_out
            )
        }, tl_suffix='ssd')
        disc_loss.backward()
        disc_optimizer.step()

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        _, disc_gen_out, disc_fmaps_real, disc_fmaps_gen = disc(real_wav, gen_wav)

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_mel=gen_mel,
                real_mel=real_mel
            ),
            'ms-stft_disc': dict(
                discs_gen_out=ssd_gen_out,
                fmaps_real=disc_fmaps_real,
                fmaps_gen=disc_fmaps_gen
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict, **disc_losses_dict}
