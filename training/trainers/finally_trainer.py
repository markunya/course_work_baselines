import torch
import torchaudio
from torch import nn
from tqdm import tqdm
from utils.model_utils import requires_grad, unwrap_model
from training.trainers.base_trainer import BaseTrainer
from transformers import WavLMModel
from training.trainers.base_trainer import gan_trainers_registry

class FinallyBaseTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.gen_name = "finally_gen"

        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            wavlm = nn.DataParallel(wavlm)
        
        self.wavlm = wavlm.to(self.device)
        self.wavlm.eval()
        requires_grad(self.wavlm, False)

    @torch.no_grad()
    def apply_wavlm(self, wav):
        if len(wav.shape) == 3:
            wav = wav.squeeze(1)

        outputs = self.wavlm(input_values=wav, output_hidden_states=True)
        features = outputs.last_hidden_state

        return features

    def synthesize_wavs(self, batch):
        gen = self.models[self.gen_name]
        gen.eval()
        
        result_dict = {
            'gen_wav': [],
            'name': []
        }

        with torch.no_grad():
            for name, input_wav in zip(batch['name'], batch['input_wav']):
                input_wav = input_wav.to(self.device)[None,None]
                wavlm_features = self.apply_wavlm(input_wav)
                gen_wav = gen(input_wav, wavlm_features).squeeze()
                
                result_dict['gen_wav'].append(gen_wav)
                result_dict['name'].append(name)
        
        result_dict['gen_wav'] = torch.stack(result_dict['gen_wav'])
        return result_dict

@gan_trainers_registry.add_to_registry(name='finally_prettrainer')
class FinallyPretrainer(FinallyBaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self):
        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        wavlm_features = self.apply_wavlm(input_wav)
        gen_wav = gen(input_wav, wavlm_features)
        gen_optimizer.zero_grad()

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_wav=gen_wav,
                real_wav=real_wav,
                wavlm=self.wavlm
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict}
    
@gan_trainers_registry.add_to_registry(name='finally_trainer')
class FinalllyTrainer(FinallyBaseTrainer):
    def __init__(self, config, sub_batch_size=None, freeze_backbone=False, n_disc_iters=2):
        super().__init__(config)

        self.backbone_freezed = not freeze_backbone
        self.n_disc_iters = n_disc_iters
        self.batch_size = config.data.train_batch_size
        self.sub_batch_size = sub_batch_size
        if self.sub_batch_size is None:
            self.sub_batch_size = self.batch_size
        
        if self.batch_size % self.sub_batch_size != 0:
            tqdm.write('Warning: sub_batch_size do not divide train_batch size.' \
            'So the total batch size with grad accumulation could be smaller than you think.')

    def _freeze_backbone_if_not_freezed(self):
        if not self.backbone_freezed:
            requires_grad(self.models[self.gen_name], False)
            requires_grad(unwrap_model(self.models[self.gen_name]).upsamplewaveunet, True)

            tqdm.write(f'Freezed {self.gen_name} backbone. Need to recreate optimizer and scheduler.')
            self.optimizers[self.gen_name] = self._create_optimizer(self.gen_name, self.config.models[self.gen_name])
            self.schedulers[self.gen_name] = self._create_scheduler(self.gen_name, self.config.models[self.gen_name])
            tqdm.write(f'Optimizer and scheduler for {self.gen_name} successfully updated')

            self.backbone_freezed = True

    def train_step(self):
        self._freeze_backbone_if_not_freezed()

        gen = self.models[self.gen_name]
        gen_optimizer = self.optimizers[self.gen_name]
        gen_loss_builder = self.loss_builders[self.gen_name]
        disc = self.models[self.disc_name]
        disc_optimizer = self.optimizers[self.disc_name]
        disc_loss_builder = self.loss_builders[self.disc_name]

        batch = next(self.train_dataloader)
        real_wav = batch['wav'].to(self.device).unsqueeze(1)
        input_wav = batch['input_wav'].to(self.device).unsqueeze(1)

        accum_steps = self.batch_size // self.sub_batch_size
        
        wavlm_features = self.apply_wavlm(input_wav)
        gen_wav = gen(input_wav, wavlm_features)

        requires_grad(disc, True)
        for _ in range(self.n_disc_iters):
            disc_optimizer.zero_grad()
            gen_wav_detached = gen_wav.detach()
            disc_losses_dict_accum = {}
            
            for i in range(accum_steps):
                start = i * self.sub_batch_size
                end = (i + 1) * self.sub_batch_size
                real_sub = real_wav[start:end]
                gen_sub = gen_wav_detached[start:end]

                disc_real_out, _, = disc(real_sub)
                disc_gen_out, _, = disc(gen_sub)

                disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
                    '': dict(
                        discs_real_out=disc_real_out,
                        discs_gen_out=disc_gen_out
                    )
                }, tl_suffix='ms-stft')

                for k, v in disc_losses_dict.items():
                    disc_losses_dict_accum[k] = disc_losses_dict_accum.get(k, 0.0) + v
                
                (disc_loss / accum_steps).backward()

            disc_optimizer.step()
            gen_losses_dict = {k: v / accum_steps for k, v in disc_losses_dict_accum.items()}

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        gen_wav_for_loss = gen_wav.detach().requires_grad_(True)
        gen_losses_dict_accum = {}

        for i in range(accum_steps):
            start = i * self.sub_batch_size
            end = (i + 1) * self.sub_batch_size
            real_sub = real_wav[start:end]
            gen_sub = gen_wav_for_loss[start:end]

            _, disc_fmaps_real = disc(real_sub)
            disc_gen_out, disc_fmaps_gen = disc(gen_sub)

            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                '': dict(
                    gen_wav=gen_sub,
                    real_wav=real_sub,
                    wavlm=self.wavlm
                ),
                'ms-stft': dict(
                    discs_gen_out=disc_gen_out,
                    fmaps_real=disc_fmaps_real,
                    fmaps_gen=disc_fmaps_gen
                )
            }, tl_suffix='gen')

            for k, v in gen_losses_dict.items():
                gen_losses_dict_accum[k] = gen_losses_dict_accum.get(k, 0.0) + v

            (gen_loss / accum_steps).backward()

        gen_wav.backward(gen_wav_for_loss.grad)
        gen_optimizer.step()

        gen_losses_dict = {k: v / accum_steps for k, v in gen_losses_dict_accum.items()}

        return {**gen_losses_dict, **disc_losses_dict}
