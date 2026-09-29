"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, Block


# ================= 1. RMSNorm =================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ================= 2. SwiGLU =================
class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ================= 3. RoPE 辅助函数 =================
def precompute_freqs_cis(dim, end, theta=10000.0):
    """预计算旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """给 Q 和 K 加上旋转位置编码"""
    # xq, xk 形状: [batch, heads, length, head_dim]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.view(1, 1, xq_.shape[2], xq_.shape[-1])

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ================= 4. 自定义 Block（应用 RoPE） =================
class MyBlock(Block):
    def __init__(self, width, heads):
        super().__init__(width, heads)
        # 替换 MLP 为 SwiGLU
        self.mlp = SwiGLU(width, 4 * width)

    def forward(self, x, freqs_cis):
        batch, length, width = x.shape
        # 1. 计算 Q, K, V (应用 RMSNorm)
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 给 Q, K 加上 RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3. 注意力计算
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(dim0=1, dim1=2).reshape(batch, length, width))

        # 4. MLP 前向传播 (应用 RMSNorm 和 SwiGLU)
        return x + self.mlp(self.norm2(x))


# ================= 5. 自定义 GPT（应用 RoPE + RMSNorm） =================
class MyGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        # 替换所有 Block 为 MyBlock
        self.blocks = nn.ModuleList([MyBlock(config['width'], config['heads']) for _ in range(config['depth'])])
        # 替换最后的 LayerNorm 为 RMSNorm
        self.norm = RMSNorm(config['width'])

        # 预计算 RoPE 的频率
        self.freqs_cis = precompute_freqs_cis(config['width'] // config['heads'], config['context'] * 2)

        # 移除原 GPT 的绝对位置编码
        self.pos = None

    def features(self, ids):
        # 重写 features
        x = self.token(ids)
        # 根据输入长度截取 RoPE 频率
        freqs_cis = self.freqs_cis[:ids.shape[1]].to(ids.device)
        for block in self.blocks:
            x = block(x, freqs_cis)
        return self.norm(x)


# ================= 6. 工厂函数 =================
def build_model(config):
    return MyGPT(config)