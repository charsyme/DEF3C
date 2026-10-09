import torch

class Spatial_Encoding(torch.nn.Module):
    def __init__(self, n_e, n_w, device, d_c, d_r, dropout=0.0, dropout_att=0.0):
        super().__init__()
        self.n_e = n_e
        self.n_w = n_w
        self.fc_layer_p = torch.nn.Linear((self.n_e + self.n_w), d_c).to(device)
        self.p_in = torch.nn.functional.one_hot(torch.arange(0, (self.n_e + self.n_w)), (self.n_e + self.n_w)).to(
            device).float()
        self.p_in = self.p_in[:(self.n_e + self.n_w), :(self.n_e + self.n_w)]
        self.norm_p = torch.nn.LayerNorm(d_c, device=device)
        self.dropout2D = torch.nn.Dropout2d(dropout)
        self.gelu = torch.nn.GELU()
        self.conv_q_e = torch.nn.Conv2d(in_channels=d_c, out_channels=d_r, kernel_size=1, device=device)
        self.conv_k_w = torch.nn.Conv2d(in_channels=d_c, out_channels=d_r, kernel_size=1, device=device)
        torch.nn.init.xavier_uniform_(self.conv_q_e.weight)
        torch.nn.init.xavier_uniform_(self.conv_k_w.weight)
        self.device = device
        self.dropout_att = dropout_att

    def forward(self):
        p = self.fc_layer_p(self.p_in)
        p = p.transpose(0, 1).unsqueeze(-1).unsqueeze(0)
        p_e = p[:, :, :self.n_e, :]
        p_w = p[:, :, self.n_e:, :]
        p_m = self.gelu(self.norm_p(p.transpose(-1, 1)).transpose(-1, 1))
        p_e_m = p_m[:, :, :self.n_e, :]
        p_w_m = p_m[:, :, self.n_e:, :]
        q_e = self.conv_q_e(p_e_m)
        k_w = self.conv_k_w(p_w_m)
        m_sp_ew = torch.matmul(q_e.transpose(1, -1), k_w.transpose(1, -1).transpose(-1, -2)) / (q_e.shape[1] ** 0.5)
        if self.training:
            msk_sp = (torch.rand([m_sp_ew.shape[-1]], device=self.device) > self.dropout_att)
            msk_sp = msk_sp.unsqueeze(0).repeat(m_sp_ew.shape[-2], 1)
            m_sp_ew = m_sp_ew.masked_fill(msk_sp.unsqueeze(0).unsqueeze(0)==0, -1e9)
        return p_e, p_w, m_sp_ew


class DiffusionEmbedding(torch.nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None, device='cpu'):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2, device=device),
            persistent=False,
        )
        self.projection1 = torch.nn.Linear(embedding_dim, projection_dim, device=device)
        self.projection2 = torch.nn.Linear(projection_dim, projection_dim, device=device)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = torch.nn.functional.gelu(x)
        x = self.projection2(x)
        #x = torch.nn.functional.gelu(x)
        return x

    def _build_embedding(self, num_steps, dim=64, device='cpu'):
        steps = torch.arange(num_steps).unsqueeze(1).to(device)  # (T,1)
        #frequencies = 10.0 ** (torch.arange(dim, device=device) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        frequencies = 10.0 ** (torch.arange(dim, device=device).float()/ (dim - 1) * 2.7).unsqueeze(0)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


def modulate(x, shift, scale):
    return x * (1 + scale.transpose(-1, -2).transpose(-2, -3)) + shift.transpose(-1, -2).transpose(-2, -3)
