from time import daylight

import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import math
import torch
import random
from torchmetrics.functional import r2_score
import copy
from model.denoising_net import Denoising_net
from model.conditioning_net import Conditioning_net
from utils.data_utils import DataLoader
from model.utils import Spatial_Encoding
from typing import Optional

class DEF_3C:
   def __init__(self, params):
       super().__init__()
       self.params = params
       self._set_diff_params()
       self.device = self.params.net.device
       self.latent_dim = self.params.net.d_channels


       self.sp_encoder = Spatial_Encoding(n_e=self.params.dataset.n_e, n_w=self.params.dataset.n_w,
                                          d_c=self.params.net.emb_channels, d_r=16,
                                          device=self.params.net.device, dropout=0.05,
                                          dropout_att=self.params.training.dropout_att)
       self.cond_net = Conditioning_net(in_channels=self.params.dataset.w_channels,
                                        t_channels =self.params.dataset.t_channels,
                                        c_channels=self.params.net.r_channels, num_blocks=self.params.net.num_ref_hgls,
                                        attn_head_dim=self.params.net.head_channels, device=self.params.net.device,
                                        emb_channels=self.params.net.emb_channels, dropout=self.params.training.dropout,

                                        dropout_att=self.params.training.dropout_att,)
       self.den_net = Denoising_net( c_channels=self.params.net.d_channels,
                                    num_blocks=self.params.net.num_den_hgls,
                                    num_steps=self.params.diffusion.num_steps, dropout=self.params.training.dropout,
                                    attn_head_dim=self.params.net.head_channels, device=self.params.net.device,
                                    emb_channels=self.params.net.emb_channels,
                                    dropout_att=self.params.training.dropout_att, dropout_cond=self.params.training.dropout_cond,
                                    )
       self.revin = RevIN(num_features=1).to(self.params.net.device)

   def _set_diff_params(self):
       if self.params.diffusion.scheduler == "linear":
           self.betas = torch.linspace(self.params.diffusion.beta_start, self.params.diffusion.beta_end,
                                       self.params.diffusion.num_steps,).to(self.params.net.device)
       elif self.params.diffusion.scheduler == "quad":
           t = torch.arange(1, self.params.diffusion.num_steps + 1).to(self.params.net.device)
           w1 = (self.params.diffusion.num_steps - t) / (self.params.diffusion.num_steps - 1)
           w2 = (t - 1) / (self.params.diffusion.num_steps - 1)
           self.betas = (w1 * self.params.diffusion.beta_start ** 0.5 + w2 *
                         self.params.diffusion.beta_end ** 0.5) ** 2
       elif self.params.diffusion.scheduler == "cosine":
           self.betas = betas_for_alpha_bar(self.params.diffusion.num_steps,
                                            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
       elif self.params.diffusion.scheduler == "sqrt_linear":
           T = self.params.diffusion.num_steps
           t = torch.arange(1, T + 1).to(self.params.net.device)
           w1 = (T - t) / (T - 1)
           w2 = (t - 1) / (T - 1)
           beta_start = self.params.diffusion.beta_start
           beta_end = self.params.diffusion.beta_end
           self.betas = (w1 * beta_start ** 0.5 + w2 * beta_end ** 0.5) ** 2
       elif self.params.diffusion.scheduler == "sqrt":
           self.betas = torch.linspace(self.params.diffusion.beta_start, self.params.diffusion.beta_end,
                                       self.params.diffusion.num_steps, dtype=torch.float64) ** 0.5
       elif self.params.diffusion.scheduler == "sigmoid":
           steepness= 10.0
           t = torch.linspace(0, 1, self.params.diffusion.num_steps)
           sigmoid = torch.sigmoid(steepness * (t - 0.5))
           self.betas = (self.params.diffusion.beta_start +
                         (self.params.diffusion.beta_end - self.params.diffusion.beta_start) * sigmoid)
           self.betas = self.betas.to(self.params.net.device)
       else:
           raise ValueError(f"schedule '{self.params.diffusion.scheduler}' unknown.")
       if not isinstance(self.betas, torch.Tensor):
           self.betas = torch.tensor(self.betas).to(self.params.net.device).to(torch.float32)
       if self.params.diffusion.rescale_betas:
           self.betas = rescale_zero_terminal_snr(self.betas)
       self.betas = self.betas.view(-1, 1, 1, 1)
       self.alphas = 1 - self.betas
       self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
       self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).view(-1, 1, 1, 1).to(self.params.net.device),
                                             self.alphas_cumprod[:-1]], dim=0)
       self.one_minus_alphas_cumprod = 1 - self.alphas_cumprod
       self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
       self.posterior_mean_coef1 = self.betas * (self.alphas_cumprod_prev ** 0.5) / (1. - self.alphas_cumprod)
       self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * (self.alphas ** 0.5) / (1. - self.alphas_cumprod)
       self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
       self.posterior_log_variance_clipped = torch.log( torch.clamp(self.posterior_variance, min=1e-20) )
       self.sigmas = (1 - self.alphas_cumprod) / self.alphas_cumprod

   def use_self_cond_at_step(self, t: int) -> bool:
       """
       Alternating proposal/refinement policy.

       Even number of steps, e.g. T=40:
           39 U -> 38 C -> ... -> 1 U -> 0 C

       Odd number of steps, e.g. T=39:
           38 U -> 37 U -> 36 C -> 35 U -> ... -> 0 C

       The two initial unconditioned steps for odd T are intentional:
       they preserve both requirements that the highest-noise state is
       unconditioned and that the final t=0 state is self-conditioned.
       """
       T = self.params.diffusion.num_steps
       return (t % 2 == 0) and (t != T - 1)

   def make_empty_self_cond(self, all_shape, device, dtype=torch.float32):
       empty = torch.zeros(
           all_shape,
           device=device,
           dtype=dtype,
       )
       return self.prepape_self_cond_v(empty, "init")

   def eval(self, split='test', seed=0, verbose=True):
       torch.manual_seed(seed)
       self.den_net.eval()
       self.cond_net.eval()
       self.sp_encoder.eval()
       eval_losses = {'mse': torch.zeros(self.params.dataset.n_e), 'rmse': torch.zeros(self.params.dataset.n_e),
                      'mae': torch.zeros(self.params.dataset.n_e), 'huber': torch.zeros(self.params.dataset.n_e),
                      'r2': torch.zeros(self.params.dataset.n_e)}
       data_loader = DataLoader(dataset=self.params.dataset.name, split=split, t_h=self.params.dataset.t_h,
                                t_f=self.params.dataset.t_f, weather_locs_sel=np.arange(self.params.dataset.n_w),
                                plant_locs_sel=np.arange(self.params.dataset.n_e), device=self.params.net.device,
                                data_dir=self.params.dataset.data_dir,)
       data_ids = data_loader.get_data_ids()
       num_batches = int(math.ceil(data_loader.get_size() / self.params.training.batch_size))
       if verbose:
           pbar = tqdm(desc="Evaluation progress", total=num_batches)
       e_preds_all, e_f_all = None, None
       timesteps = list(range(self.params.diffusion.num_steps - 1, -1, -1))  # list of ints
       with torch.no_grad():
           for batch_id in range(num_batches):
               sample_ids = data_ids[batch_id * self.params.training.batch_size:
                                     (batch_id + 1) * self.params.training.batch_size]
               [e_h_n, w_h_n, temp_h], [e_f_n, w_f_n, temp_f], time_stamp = data_loader.get_batch_data(sample_ids)
               fut_shape, obs_shape = e_f_n.shape, e_h_n.shape
               all_shape = e_h_n.shape[0], e_h_n.shape[1], e_h_n.shape[2], e_h_n.shape[3] + e_f_n.shape[3]
               sp_e, sp_w, m_sp_ew = self.sp_encoder()
               e_h_n = self.revin(e_h_n, mode='norm')

               in_cond = torch.cat([w_h_n, w_f_n], dim=-1) # B x C_w x S_w x (T_h+T_f)
               in_temp = None
               if self.params.dataset.t_channels > 0:
                   in_temp = torch.cat([temp_h, temp_f], dim=-1)
                   #in_temp = in_temp.repeat(1, 1, in_cond.shape[2], 1)

                   #in_cond = torch.cat([in_cond,
                   #                      in_temp.repeat(1, 1, in_cond.shape[-2], 1)], dim=1) # B x C_g x S_w x (T_h+T_f)
               in_obs = torch.cat([e_h_n, torch.zeros(fut_shape, device=e_h_n.device)], dim=-1)# B x 1 x S_e x (T_h+T_f)
               in_obs_padded = self.pad_energy(in_obs, -1.0, 0.0)
               noise = torch.cat([torch.zeros(obs_shape, device=in_obs.device),
                                  torch.randn(fut_shape, device=in_obs.device)], dim=-1)
               noise_padded = self.pad_energy(noise, 0.0, 1.0)
               in_den = torch.cat([in_obs_padded, noise_padded], dim=1)
               empty_self_cond = self.make_empty_self_cond(
                   all_shape,
                   device=in_den.device,
                   dtype=in_den.dtype,
               )
               self_cond = empty_self_cond

               cond_emb = self.cond_net(in_cond, sp_emb_w=sp_w, sp_emb_e=sp_e, m_ew=m_sp_ew, t=in_temp)

               for t in timesteps:
                   use_sc_now = self.use_self_cond_at_step(t)

                   out = self.den_net(
                       sample=in_den,
                       self_cond=self_cond if use_sc_now else empty_self_cond,
                       sp_emb=sp_e,
                       cond_emb=cond_emb,
                       timesteps=torch.full(
                           (in_den.shape[0],),
                           t,
                           device=in_den.device,
                           dtype=torch.long,
                       ),
                   )

                   # The proposal from t is carried only when t-1 is a
                   # conditioned refinement step.
                   carry_to_next = (
                       t > 0
                       and self.use_self_cond_at_step(t - 1)
                   )

                   in_den, self_cond = self.reverse_step(
                       in_den,
                       out,
                       t,
                       obs_shape,
                       fut_shape,
                       carry_self_cond=carry_to_next,
                   )
               e_preds_f = in_den[:, [-1], :, -2 * self.params.dataset.t_f:]
               e_preds_f = e_preds_f[:, :, :, ::2]
               e_preds_f = self.revin(e_preds_f, mode='denorm')
               e_preds = e_preds_f.squeeze(1).transpose(0, 1).detach().cpu()
               e_f = e_f_n.squeeze(1).transpose(0, 1).detach().cpu()
               if 'GEFCOM2014' in self.params.dataset.name:
                   e_preds[e_preds < 0] = 0
                   e_preds[e_preds > 1] = 1
               e_preds = e_preds.reshape(e_preds.shape[0], -1)
               e_f = e_f.reshape(e_f.shape[0], -1)
               if e_preds_all is None:
                   # First tensor defines the shape
                   e_preds_all = e_preds
                   e_f_all = e_f
               else:
                   e_preds_all = torch.cat([e_preds_all, e_preds], dim=-1)
                   e_f_all = torch.cat([e_f_all, e_f], dim=-1)
               if verbose:
                   pbar.update(1)

       mae_batch, mse_batch, rmse_batch, huber_batch, r2 = losses_cmpt(pred=e_preds_all, gt=e_f_all, delta=self.params.training.huber_delta)
       eval_losses['mae'] = mae_batch.detach().cpu().numpy()
       eval_losses['mse'] = mse_batch.detach().cpu().numpy()
       eval_losses['rmse'] = rmse_batch.detach().cpu().numpy()
       eval_losses['huber'] = huber_batch.detach().cpu().numpy()
       eval_losses['r2'] = r2.detach().cpu().numpy()
       del data_loader
       return eval_losses, e_preds_all, e_f_all


   def eval_probabilistic(
           self,
           split: str = "test",
           seed: int = 0,
           num_samples: int = 20,
           sample_chunk: int = 8,
           return_samples: bool = False,
           compute_quantiles: bool = True,
           verbose: bool = True,
   ):
       """
       Faster probabilistic evaluation.

       Main changes vs the previous version:
         1) Samples are generated in mini-ensemble chunks, e.g. 8 samples at once.
            This changes the effective network batch from B to sample_chunk * B.
         2) It does NOT store all samples for the whole test set unless return_samples=True.
         3) CRPS is computed per batch and accumulated with the correct normalization.
         4) Predictive mean/std/quantiles are still returned, but only after CPU concatenation.

       Recommended starting point:
           num_samples=20, sample_chunk=4 or 8, return_samples=False
       """
       self.den_net.eval()
       self.cond_net.eval()
       self.sp_encoder.eval()

       data_loader = DataLoader(
           dataset=self.params.dataset.name,
           split=split,
           t_h=self.params.dataset.t_h,
           t_f=self.params.dataset.t_f,
           weather_locs_sel=np.arange(self.params.dataset.n_w),
           plant_locs_sel=np.arange(self.params.dataset.n_e),
           device=self.params.net.device,
           data_dir=self.params.dataset.data_dir,
       )

       data_ids = data_loader.get_data_ids()
       num_batches = int(math.ceil(data_loader.get_size() / self.params.training.batch_size))
       timesteps = list(range(self.params.diffusion.num_steps - 1, -1, -1))
       device = self.params.net.device

       if verbose:
           pbar = tqdm(desc="Probabilistic evaluation progress", total=num_batches)

       mean_all = []
       std_all = []
       e_f_all_list = []
       q05_all, q10_all, q25_all, q50_all, q75_all, q90_all, q95_all = [], [], [], [], [], [], []
       samples_all = [] if return_samples else None

       # For normalized CRPS per series: sum_t CRPS_raw / sum_t |y_t|
       crps_num = torch.zeros(self.params.dataset.n_e, dtype=torch.float64)
       crps_den = torch.zeros(self.params.dataset.n_e, dtype=torch.float64)
       with torch.inference_mode():
           for batch_id in range(num_batches):
               sample_ids = data_ids[
                   batch_id * self.params.training.batch_size:
                   (batch_id + 1) * self.params.training.batch_size
               ]
               [e_h_n, w_h_n, temp_h], [e_f_n, w_f_n, temp_f], timestamps = data_loader.get_batch_data(sample_ids)
               single_index = timestamps[0].append(timestamps[1:])
               if batch_id == 0:
                   stamps = single_index
               else:
                   stamps = stamps.append(single_index)
               B = e_h_n.shape[0]
               fut_shape = e_f_n.shape
               obs_shape = e_h_n.shape
               all_shape = (B, e_h_n.shape[1], e_h_n.shape[2], e_h_n.shape[3] + e_f_n.shape[3])

               sp_e, sp_w, m_sp_ew = self.sp_encoder()
               e_h_n = self.revin(e_h_n, mode="norm")

               in_cond = torch.cat([w_h_n, w_f_n], dim=-1)
               in_temp = None
               if self.params.dataset.t_channels > 0:
                   in_temp = torch.cat([temp_h, temp_f], dim=-1)
                   #in_temp = in_temp.repeat(1, 1, in_cond.shape[2], 1)

               # Computed once per real batch, then repeated for ensemble chunks.
               cond_emb = self.cond_net(in_cond, sp_emb_e=sp_e, sp_emb_w=sp_w, m_ew=m_sp_ew, t=in_temp)

               # Ground truth, same layout as eval(): [S, B * Tf]
               e_f = e_f_n.squeeze(1).transpose(0, 1).detach().cpu()
               e_f = e_f.reshape(e_f.shape[0], -1)
               e_f_all_list.append(e_f)
               crps_den += e_f.abs().sum(dim=1).double()

               batch_samples = []

               remaining = num_samples
               chunk_start = 0
               while remaining > 0:
                   C = min(sample_chunk, remaining)
                   remaining -= C

                   # Expand real batch to ensemble batch: [C*B, ...]
                   e_h_rep = e_h_n.repeat(C, 1, 1, 1)
                   obs_shape_rep = (C * B, obs_shape[1], obs_shape[2], obs_shape[3])
                   fut_shape_rep = (C * B, fut_shape[1], fut_shape[2], fut_shape[3])
                   all_shape_rep = (C * B, all_shape[1], all_shape[2], all_shape[3])

                   cond_emb_rep = [cond_emb[0].repeat(C, 1, 1, 1), cond_emb[1].repeat(C, 1, 1, 1)]

                   in_obs = torch.cat([
                       e_h_rep,
                       torch.zeros(fut_shape_rep, device=device)
                   ], dim=-1)
                   in_obs_padded = self.pad_energy(in_obs, -1.0, 0.0)

                   gen = torch.Generator(device=device)
                   gen.manual_seed(seed + 100000 * batch_id + chunk_start)
                   noise_f = torch.randn(fut_shape_rep, device=device, generator=gen)
                   noise = torch.cat([
                       torch.zeros(obs_shape_rep, device=device),
                       noise_f
                   ], dim=-1)
                   noise_padded = self.pad_energy(noise, 0.0, 1.0)
                   in_den = torch.cat([in_obs_padded, noise_padded], dim=1)

                   empty_self_cond = self.make_empty_self_cond(
                       all_shape_rep,
                       device=device,
                       dtype=in_den.dtype,
                   )
                   self_cond = empty_self_cond

                   for t in timesteps:
                       use_sc_now = self.use_self_cond_at_step(t)

                       t_batch = torch.full(
                           (C * B,),
                           t,
                           device=device,
                           dtype=torch.long,
                       )

                       out = self.den_net(
                           sample=in_den,
                           self_cond=self_cond if use_sc_now else empty_self_cond,
                           sp_emb=sp_e,
                           cond_emb=cond_emb_rep,
                           timesteps=t_batch,
                       )

                       carry_to_next = (
                           t > 0
                           and self.use_self_cond_at_step(t - 1)
                       )

                       in_den, self_cond = self.reverse_step(
                           in_den,
                           out,
                           t,
                           obs_shape_rep,
                           fut_shape_rep,
                           generator=gen,
                           carry_self_cond=carry_to_next,
                       )

                   e_preds_f = in_den[:, [-1], :, -2 * self.params.dataset.t_f:]
                   e_preds_f = e_preds_f[:, :, :, ::2]

                   # Manual RevIN denorm because batch dimension was expanded from B to C*B.
                   mean_rep = self.revin.mean.repeat(C, 1, 1, 1)
                   std_rep = self.revin.stdev.repeat(C, 1, 1, 1)
                   e_preds_f = e_preds_f * std_rep + mean_rep

                   # [C*B, 1, S, Tf] -> [C, S, B*Tf]
                   e_preds = e_preds_f.view(C, B, 1, e_preds_f.shape[2], e_preds_f.shape[3])
                   e_preds = e_preds.squeeze(2).permute(0, 2, 1, 3).contiguous()
                   e_preds = e_preds.view(C, e_preds.shape[1], -1).detach().cpu()

                   if "GEFCOM2014" in self.params.dataset.name:
                       e_preds = e_preds.clamp(0.0, 1.0)

                   batch_samples.append(e_preds)
                   chunk_start += C

               # [K, S, B*Tf], only for this batch.
               batch_samples = torch.cat(batch_samples, dim=0)

               # Batch summary stats.
               mean_b = batch_samples.mean(dim=0)
               std_b = batch_samples.std(dim=0)
               mean_all.append(mean_b)
               std_all.append(std_b)

               if compute_quantiles:
                   q_levels = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
                   q = torch.quantile(batch_samples.float(), q_levels, dim=0)
                   q05_all.append(q[0])
                   q10_all.append(q[1])
                   q25_all.append(q[2])
                   q50_all.append(q[3])
                   q75_all.append(q[4])
                   q90_all.append(q[5])
                   q95_all.append(q[6])

               if return_samples:
                   samples_all.append(batch_samples)

               # Exact ensemble CRPS for this batch, accumulated raw then normalized globally.
               crps_raw = crps_ensemble_raw(batch_samples.float(), e_f.float())  # [S, B*Tf]
               crps_num += crps_raw.sum(dim=1).double()

               if verbose:
                   pbar.update(1)

       if verbose:
           pbar.close()
       del data_loader

       pred_mean = torch.cat(mean_all, dim=-1)
       pred_std = torch.cat(std_all, dim=-1)
       e_f_all = torch.cat(e_f_all_list, dim=-1)


       mae, mse, rmse, huber, r2 = losses_cmpt(
           pred=pred_mean.to(torch.float32),
           gt=e_f_all.to(torch.float32),
           delta=self.params.training.huber_delta,
       )

       eval_losses_mean = {
           "mae": mae.detach().cpu().numpy(),
           "mse": mse.detach().cpu().numpy(),
           "rmse": rmse.detach().cpu().numpy(),
           "huber": huber.detach().cpu().numpy(),
           "r2": r2.detach().cpu().numpy(),
       }

       crps_per_series = (crps_num / crps_den.clamp_min(1e-12)).detach().cpu()
       stats = {
           "mean": pred_mean,
           "std": pred_std,
           "crps": crps_per_series.mean().item(),
           "crps_per_series": crps_per_series.numpy(),
       }

       if compute_quantiles:
           q05 = torch.cat(q05_all, dim=-1)
           q10 = torch.cat(q10_all, dim=-1)
           q25 = torch.cat(q25_all, dim=-1)
           q50 = torch.cat(q50_all, dim=-1)
           q75 = torch.cat(q75_all, dim=-1)
           q90 = torch.cat(q90_all, dim=-1)
           q95 = torch.cat(q95_all, dim=-1)

           stats["q05"] = q05
           stats["q10"] = q10
           stats["q25"] = q25
           stats["q50"] = q50
           stats["q75"] = q75
           stats["q90"] = q90
           stats["q95"] = q95

           # Prediction Interval Coverage Probability and
           # Prediction Interval Normalized Average Width.
           # 50% interval: [q25, q75]
           # 80% interval: [q10, q90]
           # 90% interval: [q05, q95]
           y_range = (e_f_all.max(dim=1).values - e_f_all.min(dim=1).values).clamp_min(1e-8)
           intervals = {
               "50": (q25, q75),
               "80": (q10, q90),
               "90": (q05, q95),
           }
           for level, (q_low, q_high) in intervals.items():
               stats[f"picp{level}_per_series"] = (
                   ((e_f_all >= q_low) & (e_f_all <= q_high)).float().mean(dim=1).numpy()
               )
               stats[f"pinaw{level}_per_series"] = (
                   ((q_high - q_low).mean(dim=1) / y_range).numpy()
               )

       if return_samples:
           stats["samples"] = torch.cat(samples_all, dim=-1)

       return eval_losses_mean, stats, e_f_all, stamps

   def fit(self,):
       if not os.path.isdir(self.params.exp.checkpoints_dir):
           os.makedirs(self.params.exp.checkpoints_dir, exist_ok=True)
       file_writer, self.params.exp.log_dir = get_logger(self.params.exp.log_dir)

       huber_loss = None
       if self.params.training.use_huber:
           huber_loss = torch.nn.HuberLoss(reduction="none", delta=self.params.training.huber_delta)

       base_lr = self.params.training.l_r
       params_den = get_optimizer_param_groups(self.den_net, self.params.training.weight_decay_den, lr=base_lr)
       params_cond = get_optimizer_param_groups(self.cond_net, self.params.training.weight_decay_cond, lr=base_lr)
       params_sp = get_optimizer_param_groups(self.sp_encoder, self.params.training.weight_decay_sp, lr=base_lr)

       optimizer = torch.optim.AdamW(params_den + params_cond + params_sp, betas=(0.9, 0.999), eps=1.0e-8,
                                     lr=self.params.training.l_r,)
       lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=6, cooldown=2,
                                                                 min_lr=self.params.training.l_r * 0.01)
       earlyStopper = EarlyStopper(patience=10, min_delta=0.0)
       ema_den = EMA(self.den_net, decay=0.99)
       ema_cond = EMA(self.cond_net, decay=0.99)
       ema_sp = EMA(self.sp_encoder, decay=0.99)

       data_loader = DataLoader(dataset=self.params.dataset.name, split='train', t_h=self.params.dataset.t_h,
                                t_f=self.params.dataset.t_f, data_dir=self.params.dataset.data_dir,
                                weather_locs_sel=np.arange(self.params.dataset.n_w),
                                plant_locs_sel=np.arange(self.params.dataset.n_e),
                                device=self.params.net.device)

       data_ids = data_loader.get_data_ids()
       num_batches = int(math.ceil(data_loader.get_size() / self.params.training.batch_size))
       num_iter, sum_loss, max_epochs, copy_every = 0, 0, 500, 10

       for epoch in range(1, max_epochs):
           pbar = tqdm(desc="Training progress", total=num_batches)
           ema_den_infer = initialize_copy(self.den_net, device=self.params.net.device)
           ema_cond_infer = initialize_copy(self.cond_net, device=self.params.net.device)
           ema_sp_infer = initialize_copy(self.sp_encoder, device=self.params.net.device)
           ema_den_infer.eval()
           ema_cond_infer.eval()
           ema_sp_infer.eval()

           np.random.shuffle(data_ids)
           self.den_net.train()
           self.sp_encoder.train()
           self.cond_net.train()

           for batch_id in range(num_batches):
               sample_ids = data_ids[batch_id * self.params.training.batch_size:
                                     (batch_id + 1) * self.params.training.batch_size]
               with torch.no_grad():
                   [e_h, w_h_n, temp_h], [e_f, w_f_n, temp_f], time_stamp = data_loader.get_batch_data(sample_ids)
                   e_h_n = self.revin(e_h, mode='norm')
                   e_f_n = self.revin(e_f, mode='norm_2')
                   in_cond = torch.cat([w_h_n, w_f_n], dim=-1)
                   in_cond = augment_timeseries_2(in_cond, mult=0.1)
                   in_cond = augment_timeseries(in_cond, 0.15)

                   in_temp = None
                   if self.params.dataset.t_channels > 0:
                       in_temp = torch.cat([temp_h, temp_f], dim=-1)
                       #in_temp = in_temp.repeat(1, 1, in_cond.shape[2], 1)
                       #in_temp = augment_timeseries(in_temp)

                   e_all_n = torch.cat([e_h_n, e_f_n], dim=-1)
                   fut_shape, obs_shape, all_shape = e_f_n.shape, e_h_n.shape, e_all_n.shape
                   B_sched = all_shape[0] // 2
                   B_for = all_shape[0] - B_sched

                   # ---------------------------------------------------------
                   # One-step scheduled sampling for the alternating policy:
                   #
                   #   teacher at t + 1: unconditioned proposal
                   #   student at t:     conditioned refinement
                   #
                   # t is sampled only from actual conditioned timesteps.
                   # ---------------------------------------------------------
                   conditioned_steps = torch.tensor(
                       [
                           t for t in range(self.params.diffusion.num_steps)
                           if self.use_self_cond_at_step(t)
                       ],
                       device=self.params.net.device,
                       dtype=torch.long,
                   )

                   if conditioned_steps.numel() == 0:
                       raise ValueError(
                           "No conditioned timesteps are available. "
                           "Use at least two diffusion steps."
                       )

                   train_steps_sched = conditioned_steps[
                       torch.randint(
                           0,
                           conditioned_steps.numel(),
                           (B_sched,),
                           device=self.params.net.device,
                       )
                   ]
                   teacher_steps = train_steps_sched + 1

                   # EMA teacher conditions for the scheduled half.
                   sp_e_teacher, sp_w_teacher, m_sp_ew_teacher = ema_sp_infer()

                   if in_temp is None:
                       cond_emb_teacher = ema_cond_infer(
                           in_cond[:B_sched],
                           sp_emb_w=sp_w_teacher,
                           sp_emb_e=sp_e_teacher,
                           m_ew=m_sp_ew_teacher,
                           t=None,
                       )
                   else:
                       cond_emb_teacher = ema_cond_infer(
                           in_cond[:B_sched],
                           sp_emb_w=sp_w_teacher,
                           sp_emb_e=sp_e_teacher,
                           m_ew=m_sp_ew_teacher,
                           t=in_temp[:B_sched],
                       )

                   # Draw x_{t+1}; its teacher pass is deliberately
                   # unconditioned, matching the inference policy.
                   _, in_den_sched = self.prepare_den_forward_input(
                       e_h_n[:B_sched],
                       e_f_n[:B_sched],
                       teacher_steps,
                   )

                   teacher_empty_self_cond = self.make_empty_self_cond(
                       (
                           B_sched,
                           all_shape[1],
                           all_shape[2],
                           all_shape[3],
                       ),
                       device=in_den_sched.device,
                       dtype=in_den_sched.dtype,
                   )

                   out_sched = ema_den_infer(
                       sample=in_den_sched,
                       self_cond=teacher_empty_self_cond,
                       sp_emb=sp_e_teacher,
                       timesteps=teacher_steps,
                       cond_emb=cond_emb_teacher,
                   )

                   # One reverse transition: x_{t+1} -> x_t.
                   # The returned v/x0 proposal is carried because t is
                   # selected from the conditioned target set.
                   in_den_sched, self_cond_sched_clean = self.reverse_step(
                       in_den_sched,
                       out_sched,
                       teacher_steps,
                       [B_sched, obs_shape[1], obs_shape[2], obs_shape[3]],
                       [B_sched, fut_shape[1], fut_shape[2], fut_shape[3]],
                       sched_sampling=True,
                       carry_self_cond=True,
                   )

                   # The other half remains genuinely unconditioned.
                   unconditioned_steps = torch.tensor(
                       [
                           t for t in range(self.params.diffusion.num_steps)
                           if not self.use_self_cond_at_step(t)
                       ],
                       device=self.params.net.device,
                       dtype=torch.long,
                   )

                   train_steps_for = unconditioned_steps[
                       torch.randint(
                           0,
                           unconditioned_steps.numel(),
                           (B_for,),
                           device=self.params.net.device,
                       )
                   ]

                   _, in_den_for = self.prepare_den_forward_input(
                       e_h_n[B_sched:],
                       e_f_n[B_sched:],
                       train_steps_for,
                   )

                   self_cond_for_clean = self.make_empty_self_cond(
                       (
                           B_for,
                           all_shape[1],
                           all_shape[2],
                           all_shape[3],
                       ),
                       device=in_den_for.device,
                       dtype=in_den_for.dtype,
                   )

                   self_cond_clean = torch.cat(
                       [self_cond_sched_clean, self_cond_for_clean],
                       dim=0,
                   )

                   in_den = torch.cat(
                       [in_den_sched, in_den_for],
                       dim=0,
                   )

                   train_steps = torch.cat(
                       [train_steps_sched, train_steps_for],
                       dim=0,
                   )
               sp_e, sp_w, m_sp_ew = self.sp_encoder()
               cond_emb = self.cond_net(in_cond, sp_emb_w=sp_w,sp_emb_e=sp_e, m_ew=m_sp_ew, t=in_temp)
               out = self.den_net(
                   sample=in_den,
                   self_cond=self_cond_clean,
                   sp_emb=sp_e,
                   cond_emb=cond_emb,
                   timesteps=train_steps.to(in_den_sched.device),
               )
               in_noisy = in_den[:, [-1], :, ::2]
               target = self.x0_to_v(e_all_n, in_noisy, train_steps)

               if huber_loss is not None:
                   loss = huber_loss(out, target)
               else:
                   loss = torch.nn.functional.mse_loss(out, target, reduction="none")

               target_mask = torch.cat([torch.zeros(obs_shape), torch.ones(fut_shape)], dim=-1).to(out.device)
               loss = loss * target_mask
               loss = loss.mean(dim=(1, 2)).sum(dim=1) / target_mask.mean(dim=(1, 2)).sum(dim=1)

               if self.params.diffusion.gamma > 0.0:
                   snr = compute_snr(self.alphas_cumprod, train_steps,).to(loss.device)
                   snr_weight = torch.minimum(snr, torch.full_like(snr, self.params.diffusion.gamma,),) / (snr + 1.0)
                   loss = loss * snr_weight
               #loss[:loss.shape[0]//2] = loss[:loss.shape[0]//2] * 1.5
               loss = loss.mean()
               if self.params.training.smooth_weight > 0:
                   sm_loss = temporal_smoothness_loss(self.v_to_x0(in_noisy, train_steps, out), time_dim=-1)
                   sm_loss = sm_loss * target_mask[:, :, :, 1:]
                   sm_loss = sm_loss.mean(dim=(1, 2)).sum(dim=1) / target_mask.mean(dim=(1, 2)).sum(dim=1)
                   sm_loss = sm_loss.mean()
                   loss = loss + self.params.training.smooth_weight * sm_loss
               if self.params.training.consistency_weight > 0: # teacher prediction from self-conditioning
                   teacher = self_cond_clean[:, :, :, ::2].detach() # true training target: v / eps / x0 depending on your mode
                   target = target.detach()
                   if huber_loss is not None:
                       err_out = huber_loss(out, target)
                       err_teacher = huber_loss(teacher, target)
                   else:
                        err_out = (out - target).pow(2)
                        err_teacher = (teacher - target).pow(2) # penalize only when current prediction is worse than teacher
                   cons_loss = torch.relu(err_out - err_teacher) # optional: stronger penalty for clearly worse cases
                   cons_mask = torch.cat([ torch.ones( int(cons_loss.shape[0] // 2), cons_loss.shape[1], cons_loss.shape[2], cons_loss.shape[3], device=out.device, ), torch.zeros( int(cons_loss.shape[0] // 2), cons_loss.shape[1], cons_loss.shape[2], cons_loss.shape[3], device=out.device, ), ], dim=0)
                   cons_mask = cons_mask * target_mask
                   cons_loss = cons_loss * cons_mask
                   cons_loss = cons_loss.sum() / cons_mask.sum().clamp_min(1e-8)
                   loss = loss + self.params.training.consistency_weight * cons_loss
               optimizer.zero_grad()
               loss.backward()
               max_grad_norm = 1.0
               torch.nn.utils.clip_grad_norm_(self.den_net.parameters(), max_grad_norm)
               torch.nn.utils.clip_grad_norm_(self.cond_net.parameters(), max_grad_norm)
               torch.nn.utils.clip_grad_norm_(self.sp_encoder.parameters(), max_grad_norm)
               optimizer.step()
               ema_den.update()
               ema_cond.update()
               ema_sp.update()
               if batch_id % copy_every == 0:
                   copy_shadow_to_model(ema_den, ema_den_infer)
                   copy_shadow_to_model(ema_cond, ema_cond_infer)
                   copy_shadow_to_model(ema_sp, ema_sp_infer)

               pbar.update(1)
               num_iter = num_iter + 1
               sum_loss = sum_loss + loss.detach().cpu().numpy()
               write_logs(file_writer=file_writer, iter=num_iter, loss=sum_loss,
                          every_iters=self.params.training.log_iters)
               if num_iter > 0 and num_iter % self.params.training.log_iters == 0:
                   sum_loss = 0
           pbar.close()
           if epoch%1==0:
               snapshot_name = '{}/checkpoint_epoch_{}'.format(self.params.exp.checkpoints_dir, epoch)
               torch.save({'den_net': self.den_net.state_dict(), 'cond_net': self.cond_net.state_dict(),
                           'sp_encoder': self.sp_encoder.state_dict(),}, snapshot_name)
               ema_den.apply_shadow()
               ema_cond.apply_shadow()
               ema_sp.apply_shadow()
               val_rmse, val_huber, val_mae, val_r2 = 0, 0, 0, 0
               for eval_seed in self.params.training.eval_seeds:
                   val_losses, _, _ = self.eval_probabilistic(split='val', seed=eval_seed,verbose=True, num_samples=60, return_samples=False)
                   val_rmse = val_rmse + np.mean(val_losses['rmse'])
                   val_huber = val_huber + np.mean(val_losses['huber'])
                   val_mae = val_mae + np.mean(val_losses['mae'])
                   val_r2 = val_r2 + np.mean(val_losses['r2'])

               val_rmse = val_rmse / len(self.params.training.eval_seeds)
               val_huber = val_huber / len(self.params.training.eval_seeds)
               val_mae = val_mae / len(self.params.training.eval_seeds)
               val_r2 = val_r2 / len(self.params.training.eval_seeds)


               test_rmse, test_huber, test_mae, test_r2 = 0, 0, 0, 0
               for eval_seed in self.params.training.eval_seeds:
                   test_losses, _, _  = self.eval_probabilistic(split='test', seed=eval_seed,verbose=True, num_samples=60, return_samples=False)
                   test_rmse = test_rmse + np.mean(test_losses['rmse'])
                   test_huber = test_huber + np.mean(test_losses['huber'])
                   test_mae = test_mae + np.mean(test_losses['mae'])
                   test_r2 = test_r2 + np.mean(test_losses['r2'])

               test_rmse = test_rmse / len(self.params.training.eval_seeds)
               test_huber = test_huber / len(self.params.training.eval_seeds)
               test_mae = test_mae / len(self.params.training.eval_seeds)
               test_r2 = test_r2 / len(self.params.training.eval_seeds)


               snapshot_name = '{}/checkpoint_epoch_ema_{}'.format(self.params.exp.checkpoints_dir, epoch)
               torch.save({'den_net': self.den_net.state_dict(), 'cond_net': self.cond_net.state_dict(),
                           'sp_encoder': self.sp_encoder.state_dict(),},snapshot_name)
               ema_den.restore()
               ema_cond.restore()
               ema_sp.restore()

               print('E' + str(epoch) + 'S' + str(self.params.diffusion.num_steps))
               print('Val av. MAE : ' + str(val_mae))
               print('Val av. RMSE : ' + str(val_rmse))
               print('Val av. Huber : ' + str(val_huber))
               print('Val av. R2 : ' + str(val_r2) + '\n')

               print('Test av. MAE : ' + str(test_mae))
               print('Test av. RMSE : ' + str(test_rmse))
               print('Test av. Huber : ' + str(test_huber))
               print('Test av. R2 : ' + str(test_r2) + '\n')

               if self.params.training.use_huber:
                   lr_scheduler.step(val_huber)
                   if earlyStopper.early_stop(val_huber):
                       print("Early stopping")
                       break
               else:
                   lr_scheduler.step(val_rmse)
                   if earlyStopper.early_stop(val_rmse):
                       print("Early stopping")
                       break


   def load(self, filepath):
       checkpoint = torch.load(filepath)
       self.cond_net.load_state_dict(checkpoint['cond_net'], strict=False)
       self.den_net.load_state_dict(checkpoint['den_net'], strict=False)
       self.sp_encoder.load_state_dict(checkpoint['sp_encoder'], strict=False)

   def x0_to_v(self, x0, x_t, t):
       alpha_bar_t = self.alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
       sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
       sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
       # Estimate epsilon
       eps = (x_t - sqrt_alpha_bar_t * x0) / sqrt_one_minus_alpha_bar_t
       # Compute velocity
       v = sqrt_alpha_bar_t * eps - sqrt_one_minus_alpha_bar_t * x0
       return v

   def v_to_x0(self, x_t, t, v):
       if isinstance(t, int):
           t = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
       alpha_bar_t = self.alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1)))
       sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
       sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
       x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v
       return x0

   def prepare_den_forward_input(self, e_h_n, e_f_n, steps, augment=True):
       in_obs = torch.cat([e_h_n, torch.zeros(e_f_n.shape).to(e_f_n.device)], dim=-1)
       in_obs_padded = self.pad_energy(in_obs, -1.0, 0.0)
       noise = torch.randn(e_f_n.shape).to(e_f_n.device)
       if augment:
           noise = noise + 0.001 * torch.randn(noise.shape[0], 1, 1, 1).to(self.params.net.device)
       noise = ((self.alphas_cumprod[steps] ** 0.5) * e_f_n + (self.one_minus_alphas_cumprod[steps] ** 0.5) * noise)
       noise = torch.cat([torch.zeros(e_h_n.shape).to(noise.device), noise], dim=-1)
       noise_padded = self.pad_energy(noise, 0.0, 1.0)
       in_den = torch.cat([in_obs_padded, noise_padded], dim=1)
       return noise, in_den

   def reverse_step(
       self,
       in_den,
       out,
       t,
       obs_shape,
       fut_shape,
       sched_sampling=False,
       generator=None,
       carry_self_cond=False,
   ):
       """
       One DDPM reverse transition.

       carry_self_cond=True means that the next lower-noise step will be a
       conditioned refinement step. Only in that case are the v proposal and
       optional x0 proposal retained. Otherwise both are reset, preventing
       leakage into the following unconditioned step.
       """
       in_noisy_fut_ex = in_den[:, [-1], :, -2 * self.params.dataset.t_f:]
       in_noisy_fut = in_noisy_fut_ex[:, :, :, ::2]

       out_fut = out[:, :, :, -self.params.dataset.t_f:]
       out_x0 = self.v_to_x0(in_noisy_fut, t, out_fut)
       out_x0 = out_x0.clamp(
           -self.params.diffusion.x0_thres,
           self.params.diffusion.x0_thres,
       )

       post_mean = (
           self.posterior_mean_coef1[t] * out_x0
           + self.posterior_mean_coef2[t] * in_noisy_fut
       )

       pred_prev_sample = post_mean

       if sched_sampling or (
           isinstance(t, int) and t > 0
       ):
           if generator is not None:
               noise_extra = torch.randn(
                   fut_shape,
                   generator=generator,
                   device=out_x0.device,
               )
           else:
               noise_extra = torch.randn(
                   fut_shape,
                   device=out_x0.device,
               )

           pred_prev_sample = (
               pred_prev_sample
               + (self.posterior_variance[t] ** 0.5)
               * noise_extra
           )

       in_den_next = in_den.clone()

       # Always update the noisy future state.
       pred_prev_sample_padded = torch.ones(
           [
               pred_prev_sample.shape[0],
               pred_prev_sample.shape[1],
               pred_prev_sample.shape[2],
               2 * pred_prev_sample.shape[3],
           ],
           device=pred_prev_sample.device,
           dtype=pred_prev_sample.dtype,
       )
       pred_prev_sample_padded[:, :, :, ::2] = pred_prev_sample
       in_den_next[:, [-1], :, -2 * self.params.dataset.t_f:] = (
           pred_prev_sample_padded
       )

       # Never allow a previous x0 proposal to survive into an
       # unconditioned step.
       in_den_next[:, [-2], :, -2 * self.params.dataset.t_f:] = 0.0

       if carry_self_cond:
           if isinstance(t, int):
               t_prev = max(t - 1, 0)
           else:
               t_prev = (t - 1).clamp_min(0)

           self_cond_fut = self.x0_to_v(
               out_x0,
               pred_prev_sample,
               t_prev,
           )

           self_cond_raw = torch.cat(
               [
                   torch.zeros(
                       out_x0.shape[0],
                       out_x0.shape[1],
                       out_x0.shape[2],
                       obs_shape[-1],
                       device=out_x0.device,
                       dtype=out_x0.dtype,
                   ),
                   self_cond_fut,
               ],
               dim=-1,
           )

           self_cond_clean = self.prepape_self_cond_v(
               self_cond_raw,
               "sched",
           ).detach()

           B = out_x0.shape[0]
           device = out_x0.device

           use_x0_self_cond = torch.ones(
               B,
               dtype=torch.bool,
               device=device,
           )

           x0_idx = use_x0_self_cond.nonzero(as_tuple=True)[0]

           out_x0_padded = torch.zeros(
               out_x0.shape[0],
               out_x0.shape[1],
               out_x0.shape[2],
               2 * out_x0.shape[3],
               device=device,
               dtype=out_x0.dtype,
           )

           out_x0_padded[x0_idx, :, :, ::2] = (
               out_x0.detach()[x0_idx]
           )
           out_x0_padded[x0_idx, :, :, 1::2] = 1.0

           in_den_next[:, [-2], :, -2 * self.params.dataset.t_f:] = (
               out_x0_padded
           )

       else:
           # Empty v self-conditioning for the next unconditioned step.
           self_cond_clean = self.make_empty_self_cond(
               (
                   out_x0.shape[0],
                   out_x0.shape[1],
                   out_x0.shape[2],
                   obs_shape[-1] + fut_shape[-1],
               ),
               device=out_x0.device,
               dtype=out_x0.dtype,
           ).detach()

       return in_den_next, self_cond_clean

   def prepape_self_cond_v(self, x, mode='init'):
       if mode == 'init':
           x_new = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2 * x.shape[3]).to(torch.float32).to(x.device)
       else:
           x_new = torch.cat([torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2 * self.params.dataset.t_h),
                              torch.ones(x.shape[0], x.shape[1], x.shape[2], 2 * self.params.dataset.t_f)], dim=-1).to(torch.float32).to(x.device)
       x_new[:, :, :, ::2] = x
       return x_new

   def pad_energy(self, x, p, f):
       x_padded = torch.ones([x.shape[0], x.shape[1], x.shape[2], 2 * x.shape[3]]).to(x.device)
       x_padded[:, :, :, :2 * self.params.dataset.t_h] = p * x_padded[:, :, :, :2 * self.params.dataset.t_h]
       x_padded[:, :, :, -2 * self.params.dataset.t_f:] = f * x_padded[:, :, :, -2 * self.params.dataset.t_f:]
       x_padded[:, :, :, ::2] = x
       return x_padded


def get_logger(logging_path):
   dateTimeObj = datetime.now()
   timestampStr = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S.%f)")
   logging_path = os.path.join(logging_path, timestampStr)
   file_writer = SummaryWriter(logging_path, flush_secs=10)
   return file_writer, logging_path

def write_logs(file_writer, iter, loss, every_iters=250):
   if iter > 0 and iter % every_iters == 0:
       file_writer.add_scalar(tag="Training loss",
                              scalar_value=loss / every_iters,
                              global_step=iter)

def compute_snr(alphas_cumprod, timesteps):
   alphas_cumprod_t = alphas_cumprod[timesteps]
   snr = (alphas_cumprod_t ** 0.5) / ((1 - alphas_cumprod_t) ** 0.5 + 1e-12)
   snr = snr ** 2
   return snr.view(-1)

def losses_cmpt(pred, gt, delta):
   S, T = pred.shape
   device = pred.device
   huber_loss = torch.nn.HuberLoss( reduction='mean', delta=delta)
   mae = torch.zeros(S, device=device)
   mse = torch.zeros(S, device=device)
   rmse = torch.zeros(S, device=device)
   huber = torch.zeros(S, device=device)
   r2 = torch.zeros(S, device=device)

   for i in range(S):
       y_pred = pred[i]
       y_true = gt[i]
       mae_val = torch.nn.functional.l1_loss(y_pred, y_true, reduction='mean')
       mse_val = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean')
       rmse_val = torch.sqrt(mse_val)
       r2_val = r2_score(y_pred, y_true, multioutput='uniform_average')  # shape (1,T)
       huber_val = huber_loss(y_pred, y_true)
       mae[i] = mae_val
       mse[i] = mse_val
       rmse[i] = rmse_val
       r2[i] = r2_val
       huber[i] = huber_val
   return mae, mse, rmse, huber, r2


class RevIN(torch.nn.Module):
   def __init__(self, num_features: int, eps=1e-5):
       """
       :param num_features: the number of features or channels
       :param eps: a value added for numerical stability
       """
       super(RevIN, self).__init__()
       self.num_features = num_features
       self.eps = eps


   def forward(self, x, mode: str):
       if mode == 'norm':
           self._get_statistics(x)
           x = self._normalize(x)
       if mode == 'norm_2':
           x = self._normalize(x)
       elif mode == 'denorm':
           x = self._denormalize(x)
       return x


   def _get_statistics(self, x):
       dim2reduce = [3] # here
       self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
       self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

   def _normalize(self, x):
       x = (x - self.mean)
       x = (x / self.stdev)
       return x

   def _denormalize(self, x):
       x = (x * self.stdev)
       x = (x + self.mean)
       return x


class EarlyStopper:
   def __init__(self, patience=1, min_delta=0.0):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.min_validation_loss = float('inf')


   def early_stop(self, validation_loss):
       if validation_loss < self.min_validation_loss:
           self.min_validation_loss = validation_loss
           self.counter = 0
       elif validation_loss > (self.min_validation_loss + self.min_delta):
           self.counter += 1
           if self.counter >= self.patience:
               return True
       return False


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
   betas = []
   for i in range(num_diffusion_timesteps):
       t1 = i / num_diffusion_timesteps
       t2 = (i + 1) / num_diffusion_timesteps
       betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
   return np.array(betas)


class EMA:
   def __init__(self, model, decay=0.9999, device=None):
       self.decay = decay
       self.model = model
       self.shadow = {}
       self.device = device
       self.backup = {}
       for name, param in self.model.named_parameters():
           if param.requires_grad:
               self.shadow[name] = param.data.clone()


   def update(self):
       for name, param in self.model.named_parameters():
           if param.requires_grad:
               assert name in self.shadow
               new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
               self.shadow[name] = new_average.clone()

   def apply_shadow(self):
       self.backup = {}
       for name, param in self.model.named_parameters():
           if param.requires_grad:
               self.backup[name] = param.data.clone()
               param.data = self.shadow[name]

   def restore(self):
       for name, param in self.model.named_parameters():
           if param.requires_grad:
               param.data = self.backup[name]
       self.backup = {}


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
   """
   Rescales betas to have zero terminal SNR Based on https://huggingface.co/papers/2305.08891 (Algorithm 1)
   Args:
       betas (`torch.Tensor`):
           the betas that the scheduler is being initialized with.


   Returns:
       `torch.Tensor`: rescaled betas with zero terminal SNR
   """
   # Convert betas to alphas_bar_sqrt
   alphas = 1.0 - betas
   alphas_cumprod = torch.cumprod(alphas, dim=0)
   alphas_bar_sqrt = alphas_cumprod.sqrt()

   # Store old values.
   alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
   alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

   # Shift so the last timestep is zero.
   alphas_bar_sqrt -= alphas_bar_sqrt_T

   # Scale so the first timestep is back to the old value.
   alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

   # Convert alphas_bar_sqrt to betas
   alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
   alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
   alphas = torch.cat([alphas_bar[0:1], alphas])
   betas = 1 - alphas

   return betas


def softmin_snr_weighting(sigma):
   sigma_data = 4.0
   numerator = (sigma ** 2) * (sigma_data ** 2)
   denominator = (sigma ** 2 + sigma_data ** 2) ** 2
   return numerator / (denominator + 1e-8)  # Add epsilon for numerical safety

def get_optimizer_param_groups(model, weight_decay, lr):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no weight decay
        if (
            param.ndim == 1               # biases, norm scales
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embedding" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": lr,
        }
    ]

def temporal_smoothness_loss(y_hat: torch.Tensor, time_dim: int = -1) -> torch.Tensor:
   """
   Penalizes large differences between consecutive timesteps.
   Args:
       y_hat: tensor (..., T) where T is the time dimension.
       time_dim: which axis is the time dimension (default: last).


   Returns:
       Scalar loss (mean of squared differences).
   """
   diff = torch.diff(y_hat, dim=time_dim)   # consecutive differences
   return (diff ** 2)

def augment_timeseries(w, drop=0.25):
    x = w.clone()
    B, C, S, T = x.shape
    # apply mask -> NaN
    mask_w = (torch.rand(B, 1, S, T, device=x.device) > drop).float()
    mask_w = mask_w.repeat(1, C, 1, 1)
    x[mask_w == 0] = float("nan")
    device = x.device
    out = x.clone()

    idx = torch.arange(T, device=device, dtype=torch.float32)  # (T,)

    # Create mask of NaNs
    mask = torch.isnan(out)  # (B,C,S,T)
    known_mask = ~mask

    # Indices of last known value before each timestep
    last_known_idx = torch.zeros_like(out, dtype=torch.long)
    last_known_val = torch.zeros_like(out)

    for t in range(T):
        if t == 0:
            last_known_idx[..., t] = torch.where(known_mask[..., t], t, -1)
            last_known_val[..., t] = torch.where(known_mask[..., t], out[..., t], 0.0)
        else:
            last_known_idx[..., t] = torch.where(known_mask[..., t], t, last_known_idx[..., t - 1])
            last_known_val[..., t] = torch.where(known_mask[..., t], out[..., t], last_known_val[..., t - 1])

    # Indices of next known value after each timestep
    next_known_idx = torch.zeros_like(out, dtype=torch.long)
    next_known_val = torch.zeros_like(out)

    for t in reversed(range(T)):
        if t == T - 1:
            next_known_idx[..., t] = torch.where(known_mask[..., t], t, T)
            next_known_val[..., t] = torch.where(known_mask[..., t], out[..., t], 0.0)
        else:
            next_known_idx[..., t] = torch.where(known_mask[..., t], t, next_known_idx[..., t + 1])
            next_known_val[..., t] = torch.where(known_mask[..., t], out[..., t], next_known_val[..., t + 1])

    # Fraction for linear interpolation
    denom = (next_known_idx - last_known_idx).to(torch.float32)
    denom[denom == 0] = 1.0  # avoid div by zero
    frac = (idx.view(1, 1, 1, -1) - last_known_idx.to(torch.float32)) / denom

    interp_vals = last_known_val + frac * (next_known_val - last_known_val)

    # Fill original NaNs with interpolated values
    out[mask] = interp_vals[mask]
    return out



def augment_timeseries_2(w, mult=0.1):
    out = w * ( 1 + mult * torch.randn([w.shape[0], 1, w.shape[2], 1], device=w.device) )
    return out

def copy_shadow_to_model(ema, target_model):
    shadow = ema.shadow  # dict: name -> tensor
    tgt_state = target_model.state_dict()
    for k in tgt_state.keys():
        if k in shadow:
            tgt_state[k].copy_(shadow[k].to(tgt_state[k].device))
    target_model.load_state_dict(tgt_state)

def initialize_copy(net, device='cpu'):
    ema_net = copy.deepcopy(net)
    for p in ema_net.parameters():
        p.requires_grad = False
    ema_net.to(device)
    return ema_net


def excess_temporal_smoothness_loss(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    time_dim: int = -1,
    margin: float = 0.0,
    eps: float = 1e-8,
):

    pred_diff = torch.diff(y_pred, dim=time_dim)
    gt_diff = torch.diff(y_gt, dim=time_dim)

    pred_jump = pred_diff.abs()
    gt_jump = gt_diff.abs().detach()

    # penalize only excessive temporal variation
    excess = torch.nn.functional.relu(pred_jump - gt_jump - margin)

    loss = excess ** 2

    if mask is not None:
        mask_diff = mask[..., 1:] * mask[..., :-1]
        loss = loss * mask_diff

        denom = mask_diff.sum(dim=(1,2,3)).clamp_min(eps)
        loss = loss.sum(dim=(1,2,3)) / denom

    return loss.mean()



def crps_ensemble_raw(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Raw ensemble CRPS, no normalization.

    Args:
        samples: [K, S, T]
        y:       [S, T]
    Returns:
        crps:    [S, T]
    """
    assert samples.dim() == 3, f"Expected samples [K,S,T], got {samples.shape}"
    assert y.dim() == 2, f"Expected y [S,T], got {y.shape}"
    K, S, T = samples.shape
    assert y.shape == (S, T)

    term1 = torch.mean(torch.abs(samples - y.unsqueeze(0)), dim=0)

    xs = torch.sort(samples, dim=0).values
    i = torch.arange(1, K + 1, device=samples.device, dtype=samples.dtype).view(K, 1, 1)
    w = 2 * i - K - 1
    term2 = torch.sum(w * xs, dim=0) / (K * K)

    return term1 - term2

def crps_ensemble(samples: torch.Tensor, y: torch.Tensor, reduce: str = "none") -> torch.Tensor:
    """
    Normalized Ensemble CRPS (no mask).

    Args:
        samples: [K, S, T] ensemble samples
        y:       [S, T] observations
        reduce:  "none"  -> return [S, T] normalized
                 "mean"  -> scalar
                 "series"-> return [S] (normalized per series)

    Returns:
        Normalized CRPS
    """
    assert samples.dim() == 3, f"Expected samples [K,S,T], got {samples.shape}"
    assert y.dim() == 2, f"Expected y [S,T], got {y.shape}"

    K, S, T = samples.shape
    assert y.shape == (S, T)

    # ----- Term 1: E|X - y| -----
    term1 = torch.mean(torch.abs(samples - y.unsqueeze(0)), dim=0)  # [S,T]

    # ----- Term 2: ensemble spread term -----
    xs = torch.sort(samples, dim=0).values  # [K,S,T]
    i = torch.arange(1, K + 1, device=samples.device, dtype=samples.dtype).view(K, 1, 1)
    w = (2 * i - K - 1)
    sum_i_lt_j = torch.sum(w * xs, dim=0)  # [S,T]
    term2 = sum_i_lt_j / (K * K)

    crps = term1 - term2  # [S,T]

    # ----- Normalize like your quantile version -----
    denom_series = torch.sum(torch.abs(y), dim=1).clamp_min(1e-12)  # [S]

    if reduce == "none":
        return crps / denom_series.unsqueeze(1)  # [S,T]
    if reduce == "series":
        return torch.sum(crps, dim=1) / denom_series  # [S]
    if reduce == "mean":
        return torch.mean(torch.sum(crps, dim=1) / denom_series)

#  -----------------------------------


def quantile_loss_no_mask(target, q_pred, q):
    return 2.0 * torch.sum(
        torch.abs((q_pred - target) * ((target <= q_pred).float() - q)),
        dim=-1
    )

def calc_quantile_CRPS_no_mask(forecast, target, eps=1e-8):
    # forecast: [K, S, T]
    # target:   [S, T]
    quantiles = torch.arange(0.05, 1.0, 0.05, device=forecast.device)
    denom = torch.sum(torch.abs(target), dim=-1).clamp_min(eps)  # [S]

    crps = 0.0
    for q in quantiles:
        q_pred = torch.quantile(forecast, q.item(), dim=0)  # [S, T]
        q_loss = quantile_loss_no_mask(target, q_pred, q.item())  # [S]
        crps = crps + q_loss / denom

    return crps / len(quantiles)  # [S]

def excess_temporal_variation_score_loss(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    time_dim: int = -1,
    margin: float = 0.0,
    score: str = "rms",   # "rms" or "tv"
    eps: float = 1e-8,
):
    """
    Penalizes a prediction only when its overall temporal variation
    exceeds that of the ground truth.

    Expected shape: [B, C, K, T], with time_dim=-1.
    Returns one scalar loss averaged across B, C, K.
    """

    pred_diff = torch.diff(y_pred, dim=time_dim)
    gt_diff = torch.diff(y_gt.detach(), dim=time_dim)

    if mask is None:
        valid = torch.ones_like(pred_diff)
    else:
        # Valid only when both adjacent timesteps are forecasted/valid.
        valid = (mask[..., 1:] * mask[..., :-1]).to(pred_diff.dtype)

    denom = valid.sum(dim=time_dim).clamp_min(1.0)

    if score == "tv":
        # Mean absolute slope: closest global analogue to your current loss.
        pred_score = (pred_diff.abs() * valid).sum(dim=time_dim) / denom
        gt_score = (gt_diff.abs() * valid).sum(dim=time_dim) / denom

    elif score == "rms":
        # RMS slope: gives extra penalty to isolated large jumps/spikes.
        pred_score = torch.sqrt(
            (pred_diff.square() * valid).sum(dim=time_dim) / denom + eps
        )
        gt_score = torch.sqrt(
            (gt_diff.square() * valid).sum(dim=time_dim) / denom + eps
        )

    else:
        raise ValueError(f"Unknown score='{score}'. Use 'tv' or 'rms'.")

    excess = torch.nn.functional.relu(pred_score - gt_score - margin)

    return excess.square().mean()


import torch
import torch.nn.functional as F
from typing import Optional


def _temporal_average(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Replication-padded moving average along the last/time dimension."""
    if kernel_size <= 1:
        return x

    T = x.shape[-1]
    pad = kernel_size // 2

    z = x.reshape(-1, 1, T)
    z = F.pad(z, (pad, pad), mode="replicate")
    z = F.avg_pool1d(z, kernel_size=kernel_size, stride=1)

    return z.reshape_as(x)


def _masked_mean(x: torch.Tensor,
                 mask: Optional[torch.Tensor],
                 eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return x.mean()

    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def excess_high_frequency_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    time_dim: int = -1,
    target_smooth_kernel: int = 3,
    target_smooth_mix: float = 0.20,
    slope_weight: float = 0.10,
    curvature_weight: float = 1.00,
    slope_margin: float = 0.0,
    curvature_margin: float = 0.0,
) -> torch.Tensor:
    """
    Penalizes temporal variation and curvature only when they exceed
    a weakly smoothed target reference.

    y_hat, y: same shape, e.g. (B, C, S, T)
    mask: optional binary mask with same shape; use target/future mask.
    """

    y_hat = y_hat.movedim(time_dim, -1)
    y = y.movedim(time_dim, -1)

    if mask is not None:
        mask = mask.movedim(time_dim, -1)

    # A weakly denoised target reference.
    # mix=0 keeps the original target; larger mix assumes more target noise.
    y_smooth = _temporal_average(y, target_smooth_kernel)
    y_ref = (1.0 - target_smooth_mix) * y + target_smooth_mix * y_smooth

    # First-order changes: prevents excessively large local jumps.
    d1_hat = y_hat[..., 1:] - y_hat[..., :-1]
    d1_ref = y_ref[..., 1:] - y_ref[..., :-1]

    excess_slope = F.relu(
        d1_hat.abs() - d1_ref.abs().detach() - slope_margin
    ).square()

    # Second-order changes: specifically suppresses zig-zag / high-frequency noise.
    d2_hat = d1_hat[..., 1:] - d1_hat[..., :-1]
    d2_ref = d1_ref[..., 1:] - d1_ref[..., :-1]

    excess_curvature = F.relu(
        d2_hat.abs() - d2_ref.abs().detach() - curvature_margin
    ).square()

    if mask is not None:
        d1_mask = mask[..., 1:] * mask[..., :-1]
        d2_mask = d1_mask[..., 1:] * d1_mask[..., :-1]
    else:
        d1_mask = None
        d2_mask = None

    slope_loss = _masked_mean(excess_slope, d1_mask)
    curvature_loss = _masked_mean(excess_curvature, d2_mask)

    return slope_weight * slope_loss + curvature_weight * curvature_loss

def excess_temporal_curvature_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kernel_size: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalizes predicted temporal curvature only when it exceeds that
    of a lightly smoothed target reference. Time must be the last axis.
    """
    y_ref = _temporal_average(y_true, kernel_size=kernel_size)

    d2_hat = y_hat[..., 2:] - 2.0 * y_hat[..., 1:-1] + y_hat[..., :-2]
    d2_ref = y_ref[..., 2:] - 2.0 * y_ref[..., 1:-1] + y_ref[..., :-2]

    loss_map = F.relu(d2_hat.abs() - d2_ref.abs().detach()).square()

    if mask is None:
        return loss_map.mean()

    # All three points must belong to the forecast region.
    curvature_mask = (
        mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]
    ).to(loss_map.dtype)

    return (loss_map * curvature_mask).sum() / curvature_mask.sum().clamp_min(eps)

import torch
import torch.nn.functional as F
from typing import Optional


def daily_energy_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    day_len: int = 24,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalizes daily over-/under-production.

    Time must be the last axis. Supports inputs such as [B, C, S, T].
    For complete days, this compares total daily energy divided by 24,
    i.e. daily mean production. Dividing by 24 keeps the loss scale stable.
    """
    if y_hat.shape != y_true.shape:
        raise ValueError(
            f"y_hat and y_true must have identical shapes, got "
            f"{y_hat.shape} and {y_true.shape}."
        )
    if day_len < 1:
        raise ValueError("day_len must be at least 1.")

    if mask is None:
        mask = torch.ones_like(y_hat)
    else:
        mask = mask.to(device=y_hat.device, dtype=y_hat.dtype)
        try:
            mask = torch.broadcast_to(mask, y_hat.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"mask with shape {mask.shape} is not broadcastable "
                f"to {y_hat.shape}."
            ) from exc

    # Pad incomplete final day; padded positions receive zero mask.
    T = y_hat.shape[-1]
    pad = (-T) % day_len
    if pad:
        y_hat = F.pad(y_hat, (0, pad))
        y_true = F.pad(y_true, (0, pad))
        mask = F.pad(mask, (0, pad))

    n_days = y_hat.shape[-1] // day_len
    new_shape = (*y_hat.shape[:-1], n_days, day_len)

    y_hat = y_hat.reshape(new_shape)
    y_true = y_true.reshape(new_shape)
    mask = mask.reshape(new_shape)

    valid_count = mask.sum(dim=-1)
    valid_day = (valid_count > 0).to(y_hat.dtype)

    # Equivalent to daily total / 24 for fully observed days.
    pred_daily_energy = (y_hat * mask).sum(dim=-1) / valid_count.clamp_min(eps)
    true_daily_energy = (y_true * mask).sum(dim=-1) / valid_count.clamp_min(eps)

    daily_loss_map = (pred_daily_energy - true_daily_energy).square()

    return (
        daily_loss_map * valid_day
    ).sum() / valid_day.sum().clamp_min(1.0)

import torch


def solar_hour_importance_from_history(
    e_h: torch.Tensor,
    horizon: int,
    points_per_day: int = 24,
    start_hour: int = 0,
    q: float = 0.5,
    power: float = 1.5,
    w_min: float = 0.10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    e_h: B x 1 x S x T_h
    Returns: B x 1 x S x horizon

    Learns a soft daily solar-weight profile from historical energy.
    q=0.5 is median across historical days.
    q=0.7 gives more emphasis to the typical clear/high-generation shape.
    """
    B, C, S, T_h = e_h.shape

    if T_h < points_per_day:
        raise ValueError("History must contain at least one full day.")

    num_days = T_h // points_per_day
    usable_T = num_days * points_per_day

    # Keep only complete days, using the most recent history.
    hist = e_h[..., -usable_T:]  # B x 1 x S x (D * P)
    hist = hist.reshape(B, C, S, num_days, points_per_day)

    # Typical production at each hour of the day.
    # Median is robust to cloudy/outlier days.
    hourly_profile = torch.quantile(hist.detach(), q=q, dim=-2)
    # B x 1 x S x P

    # Normalize independently per sample and site.
    hourly_profile = hourly_profile.clamp_min(0.0)
    profile_max = hourly_profile.amax(dim=-1, keepdim=True).clamp_min(eps)
    hourly_profile = hourly_profile / profile_max

    # Align profile with the first forecast hour.
    future_hour_idx = (
        torch.arange(horizon, device=e_h.device) + start_hour
    ) % points_per_day

    importance = hourly_profile[..., future_hour_idx]
    # B x 1 x S x horizon

    # Convert [0, 1] importance into nonzero loss weights.
    weights = w_min + (1.0 - w_min) * importance.pow(power)

    return weights


import torch
import torch.nn.functional as F
from typing import Optional


def excess_temporal_smoothness_loss_global(
        y_pred: torch.Tensor,
        y_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        time_dim: int = -1,
        margin: float = 0.0,
        eps: float = 1e-8,
        aggregation: str = 'sum'
):
    # 1. Calculate step-to-step differences
    pred_diff = torch.diff(y_pred, dim=time_dim)
    gt_diff = torch.diff(y_gt, dim=time_dim)

    # 2. Get absolute jumps
    pred_jump = pred_diff.abs()
    gt_jump = gt_diff.abs().detach()

    # 3. Apply mask BEFORE aggregating over time
    if mask is not None:
        mask_diff = mask[..., 1:] * mask[..., :-1]
        pred_jump = pred_jump * mask_diff
        gt_jump = gt_jump * mask_diff

        # Calculate valid timesteps per sequence for mean calculation
        valid_steps = mask_diff.sum(dim=time_dim).clamp_min(eps)
    else:
        valid_steps = pred_jump.shape[time_dim]

    # 4. Aggregate over the entire time dimension FIRST
    if aggregation == 'sum':
        total_pred_jump = pred_jump.sum(dim=time_dim)
        total_gt_jump = gt_jump.sum(dim=time_dim)
    elif aggregation == 'mean':
        total_pred_jump = pred_jump.sum(dim=time_dim) / valid_steps
        total_gt_jump = gt_jump.sum(dim=time_dim) / valid_steps
    else:
        raise ValueError("Aggregation must be 'sum' or 'mean'")

    # 5. Apply ReLU to the aggregated totals
    # This evaluates: Did the model use a higher "wiggliness budget" than the ground truth?
    excess = F.relu(total_pred_jump - total_gt_jump - margin)

    # 6. Square the penalty
    loss = excess ** 2

    # 7. Return the mean over the remaining dimensions (Batch, Channels, etc.)
    return loss


import torch


def correlation_loss(y_pred: torch.Tensor, y_gt: torch.Tensor, time_dim: int = -1, eps: float = 1e-8):
    # 1. Center the variables by subtracting their means
    pred_mean = y_pred.mean(dim=time_dim, keepdim=True)
    gt_mean = y_gt.mean(dim=time_dim, keepdim=True)

    pred_centered = y_pred - pred_mean
    gt_centered = y_gt - gt_mean

    # 2. Calculate covariance (numerator)
    covariance = (pred_centered * gt_centered).sum(dim=time_dim)

    # 3. Calculate standard deviations (denominator)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=time_dim) + eps)
    gt_std = torch.sqrt((gt_centered ** 2).sum(dim=time_dim) + eps)

    # 4. Pearson Correlation Coefficient
    correlation = covariance / (pred_std * gt_std)

    # 5. We want to maximize correlation (make it 1), so loss is 1 - correlation
    loss = 1.0 - correlation

    return loss.mean(dim=[1, 2])