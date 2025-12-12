import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, dim, height, width, time_steps):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        self.time_steps = time_steps

        # 空间位置编码
        pos_embedding = self._get_2d_sincos_pos_embedding(dim // 2, height, width)
        self.register_buffer('pos_embedding', pos_embedding)

        # 时间位置编码
        self.time_embedding = nn.Parameter(torch.randn(1, time_steps, dim // 2))
        self.fusion = nn.Linear(dim, dim)

    def _get_2d_sincos_pos_embedding(self, embed_dim, h, w):
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)
        grid = grid.flatten(1).transpose(0, 1)

        emb_h = self._get_1d_sincos_pos_embedding(embed_dim // 2, grid[:, 0])
        emb_w = self._get_1d_sincos_pos_embedding(embed_dim // 2, grid[:, 1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    def _get_1d_sincos_pos_embedding(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.unsqueeze(1)
        out = pos * omega.unsqueeze(0)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    def forward(self, x, t_indices):
        B, T, num_patches, dim = x.shape
        pos_emb = self.pos_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        time_emb = self.time_embedding[:, :T, :].unsqueeze(2).expand(B, T, num_patches, -1)
        st_emb = torch.cat([pos_emb, time_emb], dim=-1)
        st_emb = self.fusion(st_emb)
        return st_emb


# === 核心修改区域 ===
class ST_ViT(nn.Module):
    def __init__(self,
                 input_dim=7,
                 height=10,
                 width=10,
                 time_steps=12,
                 dim=128,
                 depth=6,
                 heads=6,
                 mlp_dim=256,
                 dropout=0.1,
                 pre_len=1,
                 static_dim=4):  # <--- 新增参数: 静态特征维度
        super().__init__()

        self.patch_size = 1
        self.height = height
        self.width = width
        self.num_patches = height * width
        self.time_steps = time_steps
        self.pre_len = pre_len
        patch_dim = input_dim * self.patch_size * self.patch_size

        # Patch 嵌入
        self.patch_to_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

        # <--- 新增: 静态特征融合层
        self.use_static = static_dim > 0
        if self.use_static:
            self.static_projector = nn.Sequential(
                nn.Linear(static_dim, dim),
                nn.LayerNorm(dim),
                nn.Dropout(dropout)
            )

        self.st_embedding = SpatioTemporalEmbedding(dim, height, width, time_steps)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.time_aggregator = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 4, pre_len)
        )

    def forward(self, x, static_feat=None):  # <--- 新增参数
        # x: (B, T, C, H, W)
        # static_feat: (num_patches, static_dim)

        B, T, C, H, W = x.shape
        assert H == self.height and W == self.width, f"Input size must be {self.height}x{self.width}"

        # 动态特征处理
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x.reshape(B * T, H * W, C)
        x = self.patch_to_embedding(x)  # (B*T, num_patches, dim)
        x = x.view(B, T, self.num_patches, -1)  # (B, T, num_patches, dim)

        # <--- 新增: 融合静态特征
        if self.use_static and static_feat is not None:
            # 投影静态特征 (num_patches, dim)
            static_emb = self.static_projector(static_feat)

            # 扩展维度以适配广播: (1, 1, num_patches, dim)
            # 这样所有时间步和Batch共享同一份静态特征
            static_emb = static_emb.unsqueeze(0).unsqueeze(0)

            # 融合（相加）
            x = x + static_emb

        # 加入时空位置编码
        st_emb = self.st_embedding(x, None)
        x = x + st_emb

        # Transformer
        x = x.view(B, T * self.num_patches, -1)
        x = self.transformer(x)

        # 时间聚合
        x = x.view(B, T, self.num_patches, -1)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, T, dim)
        x = x.view(B * self.num_patches, T, -1)

        query = x[:, -1:, :]
        key_value = x
        aggregated, _ = self.time_aggregator(query, key_value, key_value)
        aggregated = aggregated.squeeze(1)

        x = aggregated.view(B, self.num_patches, -1)
        predictions = self.output_layer(x)

        predictions = predictions.view(B, H, W, self.pre_len)
        predictions = predictions.permute(0, 3, 1, 2)

        return predictions

    def c_parameters(self):
        return list(self.parameters())
