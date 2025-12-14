import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# 保留原始的 Residual、PreNorm、FeedForward、Attention、Transformer 模块（代码不变）
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
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
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

# 改进的时空位置编码
class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, dim, height, width, time_steps):
        super().__init__()
        self.dim = dim
        self.height = height
        self.width = width
        self.time_steps = time_steps
        
        # 空间位置编码（2D正弦位置编码）
        pos_embedding = self._get_2d_sincos_pos_embedding(dim // 2, height, width)
        self.register_buffer('pos_embedding', pos_embedding)  # (H*W, dim//2)
        
        # 时间位置编码（可学习）
        self.time_embedding = nn.Parameter(torch.randn(1, time_steps, dim // 2))
        
        # 融合层
        self.fusion = nn.Linear(dim, dim)

    def _get_2d_sincos_pos_embedding(self, embed_dim, h, w):
        """生成2D正弦位置编码"""
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # (2, H, W)
        grid = grid.flatten(1).transpose(0, 1)  # (H*W, 2)
        
        emb_h = self._get_1d_sincos_pos_embedding(embed_dim // 2, grid[:, 0])
        emb_w = self._get_1d_sincos_pos_embedding(embed_dim // 2, grid[:, 1])
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, embed_dim)
        return emb

    def _get_1d_sincos_pos_embedding(self, embed_dim, pos):
        """生成1D正弦位置编码"""
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.unsqueeze(1)  # (N, 1)
        out = pos * omega.unsqueeze(0)  # (N, embed_dim//2)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, embed_dim)
        return emb

    def forward(self, x, t_indices):
        """
        x: (B, T, num_patches, dim)
        t_indices: 时间步索引
        """
        B, T, num_patches, dim = x.shape
        
        # 空间位置编码
        pos_emb = self.pos_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, H*W, dim//2)
        pos_emb = pos_emb.expand(B, T, -1, -1)  # (B, T, H*W, dim//2)
        
        # 时间位置编码
        time_emb = self.time_embedding[:, :T, :]  # (1, T, dim//2)
        time_emb = time_emb.unsqueeze(2).expand(B, T, num_patches, -1)  # (B, T, H*W, dim//2)
        
        # 拼接空间和时间编码
        st_emb = torch.cat([pos_emb, time_emb], dim=-1)  # (B, T, H*W, dim)
        
        # 融合
        st_emb = self.fusion(st_emb)  # (B, T, H*W, dim)
        
        return st_emb

# 改进的 ST_ViT 模型
class ST_ViT(nn.Module):
    def __init__(self, 
                 input_dim=7,      # 输入通道数 (C)
                 height=10,        # 输入高度 (H)
                 width=10,          # 输入宽度 (W)
                 time_steps=12,    # 输入时间步数 (T)
                 dim=128,          # 嵌入维度（从64*2改为128，更清晰）
                 depth=8,          # Transformer 层数
                 heads=8,          # 注意力头数
                 mlp_dim=512,      # 前馈网络隐藏维度
                 dropout=0.1,      # Dropout率
                 pre_len=1):       # 预测时间步数 (output_steps)
        super().__init__()
        
        # Patch 参数
        self.patch_size = 1  # 每个像素作为一个 Patch
        self.height = height
        self.width = width
        self.num_patches = height * width  # 10 * 10 = 100
        self.time_steps = time_steps
        self.pre_len = pre_len
        patch_dim = input_dim * self.patch_size * self.patch_size  # patch_dim = input_dim

        # Patch 嵌入层（添加LayerNorm和Dropout）
        self.patch_to_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

        # 改进的时空位置编码
        self.st_embedding = SpatioTemporalEmbedding(dim, height, width, time_steps)

        # Transformer（添加dropout）
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)

        # 时间聚合层（使用注意力机制聚合所有时间步）
        self.time_aggregator = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        
        # 输出层（多层MLP，提升表达能力）
        self.output_layer = nn.Sequential(
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 4, pre_len)
        )

    def forward(self, x):
        # 输入形状: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert H == self.height and W == self.width, f"Input size must be {self.height}x{self.width}"

        # 处理时间维度
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x.reshape(B * T, H * W, C)  # (B*T, H*W, C) = (B*T, num_patches, input_dim)

        # Patch 嵌入
        x = self.patch_to_embedding(x)  # (B*T, num_patches, dim)

        # 恢复时间维度
        x = x.view(B, T, self.num_patches, -1)  # (B, T, num_patches, dim)

        # 添加时空位置编码
        st_emb = self.st_embedding(x, None)  # (B, T, num_patches, dim)
        x = x + st_emb  # (B, T, num_patches, dim)

        # Transformer 处理（展平时空维度）
        x = x.view(B, T * self.num_patches, -1)  # (B, T*num_patches, dim)
        x = self.transformer(x)  # (B, T*num_patches, dim)

        # 恢复时间维度以便时间聚合
        x = x.view(B, T, self.num_patches, -1)  # (B, T, num_patches, dim)
        
        # 对每个空间位置，聚合所有时间步的信息
        # 将 (B, T, num_patches, dim) 转换为 (B*num_patches, T, dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, T, dim)
        x = x.view(B * self.num_patches, T, -1)  # (B*num_patches, T, dim)
        
        # 使用最后一个时间步作为query，所有时间步作为key和value
        query = x[:, -1:, :]  # (B*num_patches, 1, dim)
        key_value = x  # (B*num_patches, T, dim)
        
        # 时间聚合
        aggregated, _ = self.time_aggregator(query, key_value, key_value)  # (B*num_patches, 1, dim)
        aggregated = aggregated.squeeze(1)  # (B*num_patches, dim)
        
        # 恢复空间维度
        x = aggregated.view(B, self.num_patches, -1)  # (B, num_patches, dim)

        # 输出预测
        predictions = self.output_layer(x)  # (B, num_patches, pre_len)

        # 恢复空间维度
        predictions = predictions.view(B, H, W, self.pre_len)  # (B, H, W, pre_len)
        predictions = predictions.permute(0, 3, 1, 2)  # (B, pre_len, H, W)

        return predictions

    def c_parameters(self):
        return list(self.parameters())  # 适配目标模型的参数分组方法
