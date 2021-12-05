import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils import checkpoint

# Conv2D + LeakyReLu + BatchNorm2D
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Conv, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        norm = nn.BatchNorm2d(out_channels, affine=True)
        self.block = nn.Sequential(conv, leaky_relu, norm)

    def forward(self, x):
        return self.block(x)

# Multi-scale Feature Extractor
class MSFE(nn.Module):
    def __init__(self, in_channels):
        super(MSFE, self).__init__()
        self.conv_1 = Conv(in_channels=in_channels, out_channels=256, kernel_size=1, padding=0)
        self.conv_3 = Conv(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv_5 = Conv(in_channels=in_channels, out_channels=256, kernel_size=5, padding=2)
        self.conv_concat = Conv(in_channels=256*3, out_channels=256, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv_concat(torch.cat([
            self.conv_1(x),
            self.conv_3(x),
            self.conv_5(x)]), dim=1)

# Auxiliary Feature Guided Self-attention
class AFGSA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False):
        super(AFGSA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)

        self.conv_map = Conv(ch*2, ch, kernel_size=1, padding=0)
        self.q_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, noisy, aux):
        n_aux = self.conv_map(torch.cat([noisy, aux], dim=1))
        b, c, h, w, block, halo, heads = *noisy.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'

        q = self.q_conv(n_aux)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q *= self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = self.k_conv(n_aux)
        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        return out

class Transformer(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, checkpoint=True):
        super(Transformer, self).__init__()
        self.checkpoint = checkpoint
        self.attention = AFGSA(ch, block_size=block_size, halo_size=halo_size, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            Conv(ch, ch, kernel_size=3, padding=1),
            Conv(ch, ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        if self.checkpoint:
            noisy = x[0] + checkpoint(self.attention, x[0], x[1])
        else:
            noisy = x[0] + self.attention(x[0], x[1])
        noisy = noisy + self.feed_forward(noisy)
        return (noisy, x[1])

class Denoiser(nn.Module):
    def __init__(self, noisy_in_ch, aux_in_ch, num_xfmr, num_gcp):
        self.noisy_msfe = MSFE(noisy_in_ch)
        self.aux_msfe = MSFE(aux_in_ch)

        xfmrs = []
        for i in range(num_xfmr):
            if i <= (num_xfmr - num_gcp):
                xfmrs.append(Transformer(ch=256, checkpoint=False))
            else:
                xfmrs.append(Transformer(ch=256, checkpoint=True))
        self.xfmrs = nn.Sequential(*xfmrs)

        self.decoder = nn.Sequential(
            Conv(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            Conv(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            Conv(in_channels=256, out_channels=3, kernel_size=3, padding=1),
        )

    def forward(self, noisy, aux):
        f_n_0 = self.noisy_msfe(noisy)
        f_a = self.aux_msfe(aux)
        f_n_k = self.xfmrs([f_n_0, f_a])[0]
        denoised = self.decoder(f_n_k) + noisy
        return denoised
        
class DiscriminatorVGG128(nn.Module):
    def __init__(self, in_ch, nf):
        super(DiscriminatorVGG128, self).__init__()

        self.features = nn.Sequential(
            Conv(in_channels=in_ch, out_channels=nf * 1, kernel_size=3, padding=1, stride=1),
            Conv(in_channels=nf * 1, out_channels=nf, kernel_size=4, padding=1, stride=2),

            Conv(in_channels=nf * 1, out_channels=nf * 2, kernel_size=3, padding=1, stride=1),
            Conv(in_channels=nf * 2, out_channels=nf * 2, kernel_size=4, padding=1, stride=2),

            Conv(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, padding=1, stride=1),
            Conv(in_channels=nf * 4, out_channels=nf * 4, kernel_size=4, padding=1, stride=2),

            Conv(in_channels=nf * 4, out_channels=nf * 8, kernel_size=3, padding=1, stride=1),
            Conv(in_channels=nf * 8, out_channels=nf * 8, kernel_size=4, padding=1, stride=2),

            Conv(in_channels=nf * 8, out_channels=nf * 8, kernel_size=3, padding=1, stride=1),
            Conv(in_channels=nf * 8, out_channels=nf * 8, kernel_size=4, padding=1, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(nf * 8 * 4 * 4, 100), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape((x.size(0), -1))
        x = self.classifier(x)
        return x

    
