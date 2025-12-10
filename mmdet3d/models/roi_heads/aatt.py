import torch
from torch import nn


class Conv2D(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, bias=False, padding=(k - 1) // 2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class AreaAttention(nn.Module):

    #yolo12
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv2D(dim, all_head_dim * 2, 1)
        self.v = Conv2D(dim, all_head_dim, 1)
        self.proj = Conv2D(all_head_dim, dim, 1)

        self.pe = Conv2D(all_head_dim, dim, 3, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)  # [B, N, C*2]
        v = self.v(x)  # [B, C, H, W]
        pp = self.pe(v)  # [B, C, H, W]
        v = v.flatten(2).transpose(1, 2)  # [B, N, C]

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        
        q, k = qk.split([C, C], dim=2)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).reshape(B, self.num_heads, self.head_dim, N)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).reshape(B, self.num_heads, self.head_dim, N)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).reshape(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        
        x = (v @ attn.transpose(-2, -1))  # [B, num_heads, head_dim, N]
        x = x.permute(0, 3, 1, 2)  # [B, N, num_heads, head_dim]

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
            
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        return self.proj(x + pp)

if __name__ == '__main__':
    model = AreaAttention(dim=64, num_heads=4, area=1).cuda()
    x = torch.randn(2, 64, 16, 16).cuda()  # [B, C, H, W]
    out = model(x) 
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")