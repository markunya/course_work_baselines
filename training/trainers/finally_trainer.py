import torch
import torchaudio
from torch import nn
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
        features = outputs.extract_features

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

@gan_trainers_registry.add_to_registry(name='finally_stage1_trainer')
class FinallyStage1Trainer(FinallyBaseTrainer):
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

@gan_trainers_registry.add_to_registry(name='finally_stage2_trainer')
class FinallyStage2Trainer(FinallyBaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.disc_name = "ms-stft_disc"

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

        wavlm_features = self.apply_wavlm(input_wav)
        gen_wav = gen(input_wav, wavlm_features)

        requires_grad(disc, True)
        for _ in range(2):
            disc_optimizer.zero_grad()

            disc_real_out, _, = disc(real_wav)
            disc_gen_out, _, = disc(gen_wav.detach())

            disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
                '': dict(
                    discs_real_out=disc_real_out,
                    discs_gen_out=disc_gen_out
                )
            }, tl_suffix='ms-stft')
            
            disc_loss.backward()
            disc_optimizer.step()

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        _, disc_fmaps_real = disc(real_wav)
        disc_gen_out, disc_fmaps_gen = disc(gen_wav)

        gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
            '': dict(
                gen_wav=gen_wav,
                real_wav=real_wav,
                wavlm=self.wavlm
            ),
            'ms-stft': dict(
                discs_gen_out=disc_gen_out,
                fmaps_real=disc_fmaps_real,
                fmaps_gen=disc_fmaps_gen
            )
        }, tl_suffix='gen')

        gen_loss.backward()
        gen_optimizer.step()

        return {**gen_losses_dict, **disc_losses_dict}

@gan_trainers_registry.add_to_registry(name='finally_stage3_trainer')
class FinallyStage3Trainer(FinallyStage2Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.backbone_freezed = False
        self.sub_batch_size = 16
        if 'train_batch_size' in config.data:
            assert config.data.train_batch_size % self.sub_batch_size == 0, \
                "Train butch size must be divisible by 16 for 3 stage trainer"

    def _freeze_backbone_if_not_freezed(self):
        if not self.backbone_freezed:
            requires_grad(self.models[self.gen_name], False)
            requires_grad(unwrap_model(self.models[self.gen_name]).upsamplewaveunet, True)
            self.backbone_freezed = True

    def _merge_loss_dicts(self, losses_list):
            out = {}
            for key in losses_list[0]:
                out[key] = sum(d[key] for d in losses_list) / len(losses_list)
            return out

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

        batch_size = real_wav.size(0)
        sub_bs = self.sub_batch_size
        num_sub_batches = batch_size // sub_bs

        all_gen_losses = []
        all_disc_losses = []

        requires_grad(disc, True)
        for _ in range(2):
            disc_optimizer.zero_grad()

            for i in range(num_sub_batches):
                real_chunk = real_wav[i * sub_bs:(i + 1) * sub_bs]
                input_chunk = input_wav[i * sub_bs:(i + 1) * sub_bs]

                with torch.no_grad():
                    wavlm_features = self.apply_wavlm(input_chunk)
                    gen_chunk = gen(input_chunk, wavlm_features)

                disc_real_out, _ = disc(real_chunk)
                disc_gen_out, _ = disc(gen_chunk.detach())

                disc_loss, disc_losses_dict = disc_loss_builder.calculate_loss({
                    '': dict(
                        discs_real_out=disc_real_out,
                        discs_gen_out=disc_gen_out
                    )
                }, tl_suffix='ms-stft')

                (disc_loss / num_sub_batches).backward()

            disc_optimizer.step()
            all_disc_losses.append(disc_losses_dict)

        requires_grad(disc, False)
        gen_optimizer.zero_grad()

        for i in range(num_sub_batches):
            real_chunk = real_wav[i * sub_bs:(i + 1) * sub_bs]
            input_chunk = input_wav[i * sub_bs:(i + 1) * sub_bs]

            wavlm_features = self.apply_wavlm(input_chunk)
            gen_chunk = gen(input_chunk, wavlm_features)

            _, disc_fmaps_real = disc(real_chunk)
            disc_gen_out, disc_fmaps_gen = disc(gen_chunk)

            gen_loss, gen_losses_dict = gen_loss_builder.calculate_loss({
                '': dict(
                    gen_wav=gen_chunk,
                    real_wav=real_chunk,
                    wavlm=self.wavlm
                ),
                'ms-stft': dict(
                    discs_gen_out=disc_gen_out,
                    fmaps_real=disc_fmaps_real,
                    fmaps_gen=disc_fmaps_gen
                )
            }, tl_suffix='gen')

            (gen_loss / num_sub_batches).backward()
            all_gen_losses.append(gen_losses_dict)

        gen_optimizer.step()

        return {
            **self._merge_loss_dicts(all_gen_losses),
            **self._merge_loss_dicts(all_disc_losses)
        }
