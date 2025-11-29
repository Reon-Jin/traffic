import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

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
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

# 修改后的 ST_ViT 模型，适配 H=W=10, patch_size=1
class ST_ViT(nn.Module):
    def __init__(self, 
                 input_dim=7,      # 输入通道数 (C)
                 height=10,        # 输入高度 (H)
                 width=10,         # 输入宽度 (W)
                 time_steps=12,    # 输入时间步数 (T)
                 dim=64*2,          # 嵌入维度
                 depth=8,          # Transformer 层数
                 heads=8,          # 注意力头数
                 mlp_dim=512,     # 前馈网络隐藏维度
                 pre_len=1):       # 预测时间步数 (output_steps)
        super().__init__()
        
        # Patch 参数
        self.patch_size = 1  # 每个像素作为一个 Patch
        self.height = height
        self.width = width
        self.num_patches = height * width  # 10 * 10 = 100

        self.pre_len=pre_len
        patch_dim = input_dim * self.patch_size * self.patch_size  # patch_dim = input_dim

        # 位置嵌入和时间嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.time_embedding = nn.Parameter(torch.randn(1, time_steps, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        # 输出层
        self.output_layer = nn.Linear(dim, pre_len)

    def forward(self, x):
        # 输入形状: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert H == self.height and W == self.width, f"Input size must be {self.height}x{self.width}"

        # 处理时间维度
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x.reshape(B * T, H * W, C)  # (B*T, H*W, C) = (B*T, num_patches, input_dim)

        # Patch 嵌入
        x = self.patch_to_embedding(x)  # (B*T, num_patches, dim)

        # 添加位置嵌入
        x += self.pos_embedding  # (B*T, num_patches, dim)

        # 恢复时间维度
        x = x.view(B, T, self.num_patches, -1)  # (B, T, num_patches, dim)

        # 添加时间嵌入
        x += self.time_embedding[:, :T, None, :]  # (B, T, num_patches, dim)

        # Transformer 处理
        x = x.view(B, T * self.num_patches, -1)  # (B, T*num_patches, dim)
        x = self.transformer(x)  # (B, T*num_patches, dim)

        # 提取最后一个时间步的表示
        x = x.view(B, T, self.num_patches, -1)  # (B, T, num_patches, dim)
        x = x[:, -1, :, :]  # (B, num_patches, dim)

        # 输出预测
        predictions = self.output_layer(x)  # (B, num_patches, pre_len)

        # 恢复空间维度
        predictions = predictions.view(B, H, W, self.pre_len)  # (B, H, W, pre_len)
        predictions = predictions.permute(0, 3, 1, 2)  # (B, pre_len, H, W)

        return predictions

    def c_parameters(self):
        return list(self.parameters())  # 适配目标模型的参数分组方法