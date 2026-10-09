import torch

class MultiheadAttention(torch.nn.Module):
    def __init__(self, dim, num_attention_heads=1, attention_head_dim=16, dropout=0.0, dropout_att=0.0, device='cpu',
                 use_mask=False):
        super(MultiheadAttention, self).__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim= attention_head_dim
        self.q = torch.nn.Conv2d(in_channels=dim, out_channels=num_attention_heads * attention_head_dim,
                                 kernel_size=(1, 3), padding=(0, 1), device=device,)
        self.k = torch.nn.Conv2d(in_channels=dim, out_channels=num_attention_heads * attention_head_dim,
                                 kernel_size=(1, 3), padding=(0, 1), device=device,)
        self.v = torch.nn.Conv2d(in_channels=dim, out_channels=num_attention_heads * attention_head_dim,
                                 kernel_size=(1, 1), padding=(0, 0), device=device,)
        self.out_proj = torch.nn.Conv2d(in_channels=num_attention_heads * attention_head_dim, out_channels=dim,
                                        kernel_size=(1, 1), padding=(0, 0), device=device,)

        self.dropout2D = torch.nn.Dropout2d(dropout)
        self.use_mask = use_mask
        if use_mask:
            self.mask = 1 - torch.triu(torch.ones([360, 360]), diagonal=1).to(device)
        self.device = device
        self.dropout_att = dropout_att

    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        # Split Q, K, V into multiple heads and then rearrange them for attention
        q = q.view(q.shape[0], self.num_attention_heads, self.attention_head_dim, q.shape[-2], q.shape[-1]).transpose(2, 3).transpose(3, 4)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(k.shape[0], self.num_attention_heads, self.attention_head_dim, k.shape[-2], k.shape[-1]).transpose(2, 3).transpose(3, 4)
        v = v.view(v.shape[0], self.num_attention_heads, self.attention_head_dim, v.shape[-2], v.shape[-1]).transpose(2, 3).transpose(3, 4)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_head_dim ** 0.5)

        if self.use_mask:
            attn_weights = attn_weights.masked_fill(self.mask==0,  -1e9)

        if self.training:
            msk_tmp = torch.rand([attn_weights.shape[-3], attn_weights.shape[-1]], device=self.device) > self.dropout_att
            msk_tmp = msk_tmp.unsqueeze(0).unsqueeze(0).unsqueeze(-2).repeat(attn_weights.shape[0], attn_weights.shape[1], 1, attn_weights.shape[-2], 1)
            attn_weights = attn_weights.masked_fill(msk_tmp==0, -1e9)

        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(3, 4).transpose(2, 3)
        attn_output = attn_output.contiguous().view(attn_output.shape[0], attn_output.shape[1] * attn_output.shape[2],
                                                    attn_output.shape[3], attn_output.shape[4])

        # Apply final linear projection
        output = self.out_proj(attn_output)
        return output
