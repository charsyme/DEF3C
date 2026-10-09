import torch
import torch.utils.checkpoint
from model.def_3c_blocks import get_block
from model.utils import DiffusionEmbedding, modulate

class Conditioning_net(torch.nn.Module):
    def __init__(self, in_channels=4, emb_channels=2, c_channels=48, attn_head_dim=16, num_blocks=1,
                 channels_group=16, dropout_att=0.0, dropout=0.1, device='cpu', t_channels=None):
        super().__init__()
        self.emb_channels = emb_channels
        self.dtype = torch.float32
        self.channel_size = c_channels
        self.act = torch.nn.GELU()
        self.dropout2d = torch.nn.Dropout2d(dropout)
        self.dropout2d_t = torch.nn.Dropout2d(dropout)

        self.blocks = torch.nn.ModuleList([])
        block_types = []
        for i in range(num_blocks):
            block_types.append("G_Block")
        for i, block_type in enumerate(block_types):
            block = get_block(block_type, channel_size=c_channels, attn_num_head_channels=attn_head_dim,
                              emb_channels=self.emb_channels, dropout=dropout, dropout_att=dropout_att,
                              channels_group=channels_group, device=device,)
            self.blocks.append(block)
        self.final_block = get_block("G_OUT_Block", channel_size=c_channels,
                                     attn_num_head_channels=attn_head_dim, emb_channels=self.emb_channels,
                                     dropout=dropout, dropout_att=dropout_att, channels_group=channels_group,
                                     device=device,)
        self.conv_in = torch.nn.Conv2d(in_channels, c_channels, (1, 1), device=device)
        self.proj_t = None
        if t_channels > 0:
            self.conv_t = torch.nn.Conv2d(t_channels, c_channels, (1, 1), device=device)
            self.proj_t = torch.nn.Linear(c_channels, emb_channels, bias=True, device=device)
            self.proj_t_2 = torch.nn.Linear(emb_channels, emb_channels, bias=True, device=device)
            self.proj_all = torch.nn.Linear(emb_channels * 2, emb_channels, bias=True, device=device)
            self.norm_t = torch.nn.LayerNorm(emb_channels, elementwise_affine=True, device=device)
            self.film_sp_t = torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(emb_channels, 2 * c_channels, bias=True, device=device)
            )
            torch.nn.init.zeros_(self.film_sp_t[-1].weight)
            torch.nn.init.zeros_(self.film_sp_t[-1].bias)

        self.film_sp = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(emb_channels, 2 * c_channels, bias=True, device=device)
        )

        torch.nn.init.zeros_(self.film_sp[-1].weight)
        torch.nn.init.zeros_(self.film_sp[-1].bias)
        self.norm_sp = torch.nn.LayerNorm(c_channels, elementwise_affine=False, device=device)
        self.proj_sp = torch.nn.Conv2d(c_channels, c_channels, (1, 1), device=device)
        self.t_drop = 0.00

    def forward(self, x, t=None, m_ew=None, sp_emb_w=None, sp_emb_e=None):
        x = self.conv_in(x)
        #sp_emb = self.proj_sp(self.act(sp_emb))
        #x = x + sp_emb
        shift_cond, scale_cond = self.film_sp(sp_emb_w.transpose(1, 2).transpose(2, 3)).chunk(2, dim=-1)
        x = modulate(self.norm_sp(x.transpose(1, -1)).transpose(1, -1), shift_cond, scale_cond)
        x = self.proj_sp(x)

        for i, block in enumerate(self.blocks):
            x = block(x=x)
        x = self.final_block(x=x, m_ew=m_ew)
        y = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[3]], device=x.device)

        if self.proj_t is not None:
            if self.training and torch.rand(1).item() < self.t_drop:
                y = torch.zeros([t.shape[0], x.shape[1], t.shape[2], t.shape[3]], device=x.device)
            else:
                t = self.conv_t(t)
                shift_cond_t, scale_cond_t = self.film_sp_t(sp_emb_e.transpose(1, 2).transpose(2, 3)).chunk(2, dim=-1)
                t = modulate(self.norm_sp(t.transpose(1, -1)).transpose(1, -1), shift_cond_t, scale_cond_t)
                t = self.proj_t(t.transpose(1, 2).transpose(2, 3))
                y = self.norm_t(t).transpose(2, 3).transpose(1, 2)
            #x = self.act(torch.cat([x, t], dim=1))
            #x = self.proj_all(x.transpose(1, 2).transpose(2, 3)).transpose(2, 3).transpose(1, 2)
        return [x, y]
