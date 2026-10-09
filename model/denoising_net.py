import torch
from model.def_3c_blocks import get_block
from model.utils import DiffusionEmbedding, modulate


class Denoising_net(torch.nn.Module):
    def __init__(
        self,
        out_channels=1,
        num_blocks=2,
        c_channels=48,
        attn_head_dim=16,
        num_steps=10,
        emb_channels=12,
        device="cpu",
        dropout=0.1,
        dropout_att=0.0,
        dropout_cond = 0.0,
        channels_group=16,
        forecast_horizon=48,
    ):
        super().__init__()

        self.emb_channels = emb_channels
        self.channel_size = c_channels
        self.forecast_horizon = forecast_horizon
        self.self_width = c_channels // 2

        # ------------------------------------------------------------------
        # Main denoising backbone
        # ------------------------------------------------------------------
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                get_block(
                    "E_BLOCK",
                    channel_size=c_channels,
                    attn_num_head_channels=attn_head_dim,
                    emb_channels=emb_channels,
                    dropout=dropout,
                    dropout_att=dropout_att,
                    device=device,
                    channels_group=channels_group,
                )
            )

        # Input consists of:
        # [past energy with future masked,
        #  noisy sample,
        #  future energy with history masked]
        self.proj_e = torch.nn.Conv2d(
            3,
            c_channels,
            kernel_size=(1, 2),
            stride=(1, 2),
            device=device,
        )

        self.proj_sp = torch.nn.Conv2d(
            c_channels,
            c_channels,
            kernel_size=1,
            device=device,
        )

        self.proj_head = torch.nn.Conv2d(
            c_channels,
            self.self_width,
            kernel_size=1,
            device=device,
        )

        self.proj_out = torch.nn.Conv2d(
            self.self_width,
            out_channels,
            kernel_size=1,
            device=device,
        )

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=emb_channels,
            device=device,
        )

        # ------------------------------------------------------------------
        # Main conditioning
        # ------------------------------------------------------------------
        self.film_sp = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(
                emb_channels,
                2 * c_channels,
                bias=True,
                device=device,
            ),
        )

        self.film_head = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(
                emb_channels,
                2 * c_channels,
                bias=True,
                device=device,
            ),
        )

        self.norm_sp = torch.nn.LayerNorm(
            c_channels,
            elementwise_affine=False,
            device=device,
        )

        self.norm_head = torch.nn.LayerNorm(
            c_channels,
            elementwise_affine=False,
            device=device,
        )

        # ------------------------------------------------------------------
        # Self-conditioning relation branch
        # self_cond is expected as [B, 1, S, 2T]:
        # interleaved [self-condition value, availability mask].
        # ------------------------------------------------------------------
        self.self_proj = torch.nn.Conv2d(
            1,
            emb_channels,
            kernel_size=(1, 2),
            stride=(1, 2),
            device=device,
        )

        self.self_norm = torch.nn.LayerNorm(
            self.self_width,
            elementwise_affine=False,
            device=device,
        )

        self.self_z_norm = torch.nn.LayerNorm(
            emb_channels,
            elementwise_affine=False,
            device=device,
        )

        # Used only by the acceptance gate.
        self.self_h_proj = torch.nn.Conv2d(
            self.self_width,
            self.self_width,
            kernel_size=1,
            device=device,
        )

        self.self_z_rel_proj = torch.nn.Conv2d(
            emb_channels,
            self.self_width,
            kernel_size=1,
            device=device,
        )

        # z -> [shift, scale], both with self_width channels.
        self.self_film = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(
                emb_channels,
                2 * self.self_width,
                bias=True,
                device=device,
            ),
        )

        # Shared proposal network.
        # It is evaluated both before and after self-conditioning FiLM.
        self.self_stream_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.self_width,
                self.self_width,
                kernel_size=1,
                device=device,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                self.self_width,
                self.self_width,
                kernel_size=1,
                device=device,
            ),
        )

        # [h_p, z_p, proposal, availability] -> per-channel acceptance.
        self.self_accept_gate = torch.nn.Conv2d(
            3 * self.self_width + 1,
            self.self_width,
            kernel_size=1,
            device=device,
        )

        self.dropout2d = torch.nn.Dropout2d(dropout)
        self.dropout2d_cond = torch.nn.Dropout2d(dropout_cond)



        # ------------------------------------------------------------------
        # Zero-initialized adaptive modulation heads.
        # ------------------------------------------------------------------
        torch.nn.init.zeros_(self.film_sp[-1].weight)
        torch.nn.init.zeros_(self.film_sp[-1].bias)

        torch.nn.init.zeros_(self.film_head[-1].weight)
        torch.nn.init.zeros_(self.film_head[-1].bias)

        torch.nn.init.zeros_(self.self_film[-1].weight)
        torch.nn.init.zeros_(self.self_film[-1].bias)

        # Initially gives a conservative constant gate.
        torch.nn.init.zeros_(self.self_accept_gate.weight)
        torch.nn.init.constant_(self.self_accept_gate.bias, -2.0)

    @staticmethod
    def _layer_norm_channels(x, norm):
        """
        x: [B, C, S, T]
        LayerNorm acts over C, so move channels last temporarily.
        """
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, sample, timesteps, sp_emb=None, self_cond=None, cond_emb=None):
        cond_emb_w, cond_emb_t = cond_emb

        # ------------------------------------------------------------------
        # Input construction
        # ------------------------------------------------------------------
        x_hist = sample[:, [0], :, :].clone()
        x_hist[:, :, :, -self.forecast_horizon:] = 0.0

        x_noisy = sample[:, [1], :, :]

        x_future = sample[:, [0], :, :].clone()
        x_future[:, :, :, :-self.forecast_horizon] = 0.0

        x = torch.cat([x_hist, x_noisy, x_future], dim=1)
        x = self.proj_e(x)

        # ------------------------------------------------------------------
        # Spatial conditioning
        # ------------------------------------------------------------------
        shift_sp, scale_sp = self.film_sp(
            sp_emb.permute(0, 2, 3, 1)
        ).chunk(2, dim=-1)

        x = modulate(
            self._layer_norm_channels(x, self.norm_sp),
            shift_sp,
            scale_sp,
        )
        x = self.proj_sp(x)

        # ------------------------------------------------------------------
        # Diffusion and external conditioning
        # ------------------------------------------------------------------
        tim_emb = self.diffusion_embedding(timesteps)
        tim_emb_4d = tim_emb.unsqueeze(-1).unsqueeze(-1)

        emb_block_cond = self.dropout2d_cond(cond_emb_w) + tim_emb_4d
        if cond_emb_t is not None and cond_emb_t.norm().sum() > 0:
            emb_block_cond = self.dropout2d_cond(cond_emb_t) + emb_block_cond

        for block in self.blocks:
            x = block(
                x=x,
                emb=emb_block_cond,
                tim_emb=tim_emb_4d,
            )

        # ------------------------------------------------------------------
        # Head conditioning
        # ------------------------------------------------------------------
        shift_head, scale_head = self.film_head(
            emb_block_cond.permute(0, 2, 3, 1)
        ).chunk(2, dim=-1)

        x = modulate(
            self._layer_norm_channels(x, self.norm_head),
            shift_head,
            scale_head,
        )
        x = self.proj_head(x)

        # ------------------------------------------------------------------
        # Self-conditioning relation branch
        # ------------------------------------------------------------------
        if self_cond is not None:

            # [B, 1, S, T], after removing interleaving.
            self_mask = self_cond[:, :, :, 1::2].to(dtype=x.dtype).clamp(0.0, 1.0)

            if self_mask.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    "self_cond mask shape must match the projected feature map. "
                    f"Got self_mask {tuple(self_mask.shape)} and x {tuple(x.shape)}."
                )

            # Encode self-conditioning pairs and make the embedding timestep-aware.
            z = self.self_proj(self_cond)

            # Independently normalize the internal current state and modality.
            h_n = self._layer_norm_channels(x, self.self_norm)
            z_n = self._layer_norm_channels(z, self.self_z_norm)

            # These remain separate learned representations for relational gating.
            h_p = self.self_h_proj(h_n)
            z_p = self.self_z_rel_proj(z_n)

            # Self-conditioning generates FiLM parameters.
            self_shift, self_scale = self.self_film(
                z_n.permute(0, 2, 3, 1)
            ).chunk(2, dim=-1)

            self_shift = self_shift.permute(0, 3, 1, 2)
            self_scale = self_scale.permute(0, 3, 1, 2)

            # FiLM is applied to normalized current features.
            h_mod = h_n * (1.0 + self_scale) + self_shift

            # The proposal is specifically the effect of self-conditioning.
            # Both terms use the same learned transformation, so this does not
            # assume that x and self_cond share a raw feature space.
            proposal = (self.self_stream_mlp(h_mod) - self.self_stream_mlp(h_n))

            # The current representation evaluates the proposed update.
            gate_input = torch.cat([h_p, z_p, proposal, self_mask],dim=1,)

            accept_gate = torch.sigmoid( self.self_accept_gate(gate_input) )

            #correction = 0.05 * torch.tanh(proposal)

            x = x + self_mask * self.dropout2d(accept_gate * proposal)

        x = self.proj_out(x)
        return x