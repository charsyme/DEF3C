import torch
from model.attention import MultiheadAttention
from model.unet import UNet
from model.utils import modulate

def get_block(down_block_type, channel_size, emb_channels, attn_num_head_channels=16,
              device='cpu', dropout=0.1, dropout_att=0.1, channels_group=16):
    if down_block_type == "G_OUT_Block":
        return G_out_block(channel_size=channel_size, emb_channels=emb_channels, device=device)
    elif down_block_type == "E_BLOCK":
        return E_Block(channel_size=channel_size, device=device, dropout=dropout, emb_channels=emb_channels,
                       attn_num_head_channels=attn_num_head_channels, dropout_att=dropout_att,)
    elif down_block_type == "G_Block":
        return G_block(channel_size=channel_size, device=device, dropout=dropout, dropout_att=dropout_att,
                       attn_num_head_channels=attn_num_head_channels, channels_group=channels_group)
    raise ValueError(f"{down_block_type} does not exist.")


class G_out_block(torch.nn.Module):
    def __init__(self, channel_size, emb_channels, device='cpu'):
        super().__init__()
        self.emb_channels = emb_channels
        self.proj_cond = torch.nn.Conv2d(channel_size, emb_channels, (1, 1), device=device)
        self.norm_cond = torch.nn.LayerNorm(channel_size, device=device, eps=1e-06)
        self.norm_out = torch.nn.LayerNorm(emb_channels, device=device, eps=1e-06)
        self.act = torch.nn.GELU()

    def forward(self, x, m_ew):
        y = self.norm_cond(x.transpose(1, -1)).transpose(1, -1)
        y = torch.matmul(torch.softmax(m_ew, dim=-1), y)
        y = self.proj_cond(y)
        y = self.norm_out(y.transpose(1, -1)).transpose(1, -1)
        return y

class G_block(torch.nn.Module):
    def __init__(self, channel_size, attn_num_head_channels=16, dropout_att=0.0, dropout=0.0, channels_group=16,
                 device='cpu'):
        super().__init__()
        self.hourglass = UNet(in_channels=channel_size, dropout=dropout, device=device,
                              features=[channel_size, channel_size, channel_size],)
        self.dropout2d = torch.nn.Dropout2d(dropout)
        self.norm_qkv = torch.nn.LayerNorm(channel_size, device=device)
        self.attention = MultiheadAttention(dim=channel_size, attention_head_dim=attn_num_head_channels,
                                            num_attention_heads=channel_size // attn_num_head_channels,
                                            dropout_att=dropout_att, dropout=dropout, device=device, )

    def forward(self, x):
        y = x + self.dropout2d(self.hourglass(x))
        qkv = self.norm_qkv(y.transpose(-1, 1)).transpose(-1, 1)
        y = y + self.dropout2d(self.attention(query=qkv, key=qkv, value=qkv))
        return y


class E_Block(torch.nn.Module):
    def __init__(self, channel_size, emb_channels, dropout=0.1, dropout_att=0.0,
                 attn_num_head_channels=1, device='cpu'):
        super().__init__()
        self.attn_num_head_channels = attn_num_head_channels

        self.hourglass = UNet(in_channels=channel_size, ada=True, device=device, dropout=dropout,
                              emb_channels=emb_channels, features=[channel_size, channel_size, channel_size],)
        self.dropout2d = torch.nn.Dropout2d(dropout)
        self.attention = MultiheadAttention(dim=channel_size, device=device, attention_head_dim=attn_num_head_channels,
                                            dropout=dropout, dropout_att=dropout_att, use_mask=True,
                                            num_attention_heads=channel_size // attn_num_head_channels)
        self.film_params = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(emb_channels, 6 * channel_size, bias=True, device=device)
        )
        self.norm_qkv = torch.nn.LayerNorm(channel_size, elementwise_affine=False, device=device)
        self.norm_hour = torch.nn.LayerNorm(channel_size, elementwise_affine=False, device=device)
        torch.nn.init.zeros_(self.film_params[-1].weight)
        torch.nn.init.zeros_(self.film_params[-1].bias)

    def forward(self, x, emb=None, tim_emb=None):
        B, channel, K, L = x.shape
        shift_h, scale_h, gate_h, shift_att, scale_att, gate_att = self.film_params(emb.transpose(1, 2).transpose(2, 3)).chunk(6, dim=-1)

        y_hour = modulate(self.norm_hour(x.transpose(1, -1)).transpose(1, -1), shift_h, scale_h)
        y_hour = self.hourglass(y_hour, emb_t=tim_emb)
        y = x + self.dropout2d(gate_h.transpose(-2, -1).transpose(-2, -3) * y_hour)
        y_in = self.norm_qkv(y.reshape(B, channel, K, L).transpose(-1, 1)).transpose(-1, 1)
        y_in = modulate(y_in, shift_att, scale_att)
        y_att = self.attention(query=y_in, key=y_in, value=y_in)
        y = y + self.dropout2d(gate_att.transpose(-2, -1).transpose(-2, -3) * y_att)
        return y.reshape(B, channel, K, L)
