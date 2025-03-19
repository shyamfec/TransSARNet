# importing required libraries
import math
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head,
        heads
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)

        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        # scale

        q *= self.scale

        # attention

        sim = einsum('b i d, b j d -> b i j', q, k)

        # add relative positional bias

        sim += self.rel_pos_emb(q)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge and combine heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)

        # merge blocks back to original feature map

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        return out
        
def sobel_kernel():
    # Define the Sobel kernels (for horizontal and vertical edge detection)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
                        
    sobel_left = np.array([[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]], dtype=np.float32)
                        
    sobel_right = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]], dtype=np.float32)
    
    # Convert to PyTorch tensors and stack the kernels
    return np.stack([sobel_x, sobel_y, sobel_left, sobel_right], axis=0)  # Stack both kernels for multi-channel case 
    
class TrainableSobelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(TrainableSobelConv2d, self).__init__()
        
        # Initialize the kernel with Sobel filters (for edge detection)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize the filters with Sobel kernels
        sobel_filters = sobel_kernel()
        sobel_filters = torch.tensor(sobel_filters, dtype=torch.float32).unsqueeze(1)  # [2, 1, 3, 3]
        
        # Make it trainable by using nn.Parameter
        self.filters = nn.Parameter(sobel_filters)
        
        # Optionally, define a bias term if needed
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Perform the convolution with the trainable filters
        return F.conv2d(x, self.filters, self.bias, self.stride, self.padding)

class TrainableMeanFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TrainableMeanFilter, self).__init__()
        
        # Initialize the kernel to represent a mean filter
        self.kernel_size = kernel_size
        
        # Define the convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        
        # Initialize the kernel to the mean filter values
        with torch.no_grad():
            # Create a mean filter with equal values
            mean_filter = torch.ones((out_channels, in_channels, kernel_size, kernel_size)) / (kernel_size * kernel_size)
            self.conv.weight.copy_(mean_filter)
    
    def forward(self, x):
        return self.conv(x)

class featureDenoiseBlk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(featureDenoiseBlk, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect')
        self.activ1 = nn.PReLU()
        self.mean_filter = TrainableMeanFilter(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect')
    
    def forward(self, x):
        x1 = self.activ1(self.conv1(x))
        x2 = self.mean_filter(x1)
        x3 = self.conv2(x2)
        out = torch.add(x1,x3)
        return out
 
class textureMapHFBlock(nn.Module):
    def __init__(self,inChnl, outChnl):
        super(textureMapHFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = inChnl, out_channels = 16, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect')
        self.activ1 = nn.ReLU()
        self.BNconv1 = nn.BatchNorm2d(16)
        self.avgPool1 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 2)
        self.activ2 = nn.ReLU()
        self.BNconv2 = nn.BatchNorm2d(8)
        self.avgPool2 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 2)
        self.activ3 = nn.ReLU()
        self.BNconv3 = nn.BatchNorm2d(8)
        self.avgPool3 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (5,1), dilation = 1, stride = 1, padding = (2,0), padding_mode = 'reflect', groups = 2)
        self.activ4 = nn.ReLU()
        self.BNconv4 = nn.BatchNorm2d(8)
        self.avgPool4 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1,5), dilation = 1, stride = 1, padding = (0,2),padding_mode = 'reflect', groups = 2)
        self.activ5 = nn.ReLU()
        self.BNconv5 = nn.BatchNorm2d(8)
        self.avgPool5 = nn.AdaptiveAvgPool2d((64,64))        
        
        self.conv6 = nn.Conv2d(in_channels = 32, out_channels = outChnl, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 1)        
        self.avgPool6 = nn.AdaptiveAvgPool2d((64,64))      
    def forward(self,x):
        x1 = self.avgPool1(self.BNconv1(self.activ1(self.conv1(x))))        
        x2 = self.avgPool2(self.BNconv2(self.activ2(self.conv2(x1))))
        x3 = self.avgPool3(self.BNconv3(self.activ3(self.conv3(x1))))
        x4 = self.avgPool4(self.BNconv4(self.activ4(self.conv4(x1))))
        x5 = self.avgPool5(self.BNconv5(self.activ5(self.conv5(x1))))        
        x3_cat = torch.cat((x2,x3,x4,x5), dim = 1)        
        x6 = self.avgPool6(self.conv6(x3_cat))
        return x6

class textureMapHF(nn.Module):
    def __init__(self,inChnl,outChnl, num_blocks) -> None:
        super(textureMapHF, self).__init__()
        self.blocks = nn.ModuleList(
                      [textureMapHFBlock(inChnl[i],outChnl[i]) for i in range(num_blocks)])
        self.layer_norm = nn.LayerNorm([64,64])
              
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)               
        return x 

class textureMapLF(nn.Module):
    def __init__(self, inChnl):
        super(textureMapLF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = inChnl, out_channels = 64, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect')
        self.activ1 = nn.ReLU()
        self.BNconv1 = nn.BatchNorm2d(64)
        self.avgPool1 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 4)
        self.activ2 = nn.ReLU()
        self.BNconv2 = nn.BatchNorm2d(32)
        self.avgPool2 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 4)
        self.activ3 = nn.ReLU()
        self.BNconv3 = nn.BatchNorm2d(16)
        self.avgPool3 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 4)
        self.activ4 = nn.ReLU()
        self.BNconv4 = nn.BatchNorm2d(8)
        self.avgPool4 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv5 = nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 4)
        self.activ5 = nn.ReLU()
        self.BNconv5 = nn.BatchNorm2d(4)
        self.avgPool5 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv6 = nn.Conv2d(in_channels = 4, out_channels = 2, kernel_size = 3, dilation = 2, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 2)
        self.activ6 = nn.ReLU() 
        self.BNconv6 = nn.BatchNorm2d(2)
        self.avgPool6 = nn.AdaptiveAvgPool2d((64,64))
        
        self.conv7 = nn.Conv2d(in_channels = 2, out_channels = 64, kernel_size = 3, dilation = 1, stride = 1, padding = 'same',padding_mode = 'reflect', groups = 1)
        self.activ7 = nn.ReLU()
        self.BNconv7 = nn.BatchNorm2d(64)
        self.avgPool7 = nn.AdaptiveAvgPool2d((64,64))
                
    def forward(self,x):
        x1 = self.avgPool1(self.BNconv1(self.activ1(self.conv1(x))))       
        x2 = self.avgPool2(self.BNconv2(self.activ2(self.conv2(x1))))
        x3 = self.avgPool3(self.BNconv3(self.activ3(self.conv3(x2))))        
        x4 = self.avgPool4(self.BNconv4(self.activ4(self.conv4(x3))))
        x5 = self.avgPool5(self.BNconv5(self.activ5(self.conv5(x4))))        
        x6 = self.avgPool6(self.BNconv6(self.activ6(self.conv6(x5))))
        out = self.avgPool7(self.BNconv7(self.activ7(self.conv7(x6))))           
        return out        
 
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super(PositionwiseFeedForward,self).__init__()
        self.pffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff,d_model)
        )        
    def forward(self, x: Tensor) -> Tensor:
        return self.pffn(x)
        
class EncoderBlock(nn.Module):
    def __init__(self, img_size, d_model, block_size, halo_size, dim_head, heads, d_ff, dropout) -> None:
        super(EncoderBlock,self).__init__()
        self.self_attn = HaloAttention(dim = d_model, block_size=block_size, halo_size=halo_size, dim_head = dim_head, heads = heads)        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm([img_size,img_size])
        self.norm2 = nn.LayerNorm([img_size,img_size])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:        
        attn_output = self.self_attn(x)        
        x = self.norm1(x + self.dropout(attn_output))        
        xin_pffn = rearrange(x,'b c h w -> b (h w) c')
        ff_output = self.feed_forward(xin_pffn)        
        ff_output = rearrange(ff_output,'b (h w) c -> b c h w', h = int(math.sqrt(ff_output.shape[1])), w = int(math.sqrt(ff_output.shape[1])))        
        x = self.norm2(x + self.dropout(ff_output))
        return x   
      
class Transformer(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        
        self.tmapHF = textureMapHF(inChnl = [1,32,64,64], outChnl = [32,64,64,64], num_blocks = 4)
        self.tmapLF = textureMapLF(inChnl = 1)
        self.convFM = nn.Conv2d(in_channels = 64+64+4, out_channels = 1, kernel_size = 1, dilation = 1, stride = 1, padding = 'same')
        self.reluFM = nn.ReLU()
        
        self.sobel_conv = TrainableSobelConv2d(1,4)
        
        self.fdb1 = featureDenoiseBlk(64+64+4,64+64+4)
        self.fracPool1 = nn.FractionalMaxPool2d(3, output_size=(32, 32))
        self.encoder1 = EncoderBlock(64, 64+64+4, 4, 2, 16, 4, 660, 0.2) #d_model, block_size, halo_size, dim_head, heads, d_ff
        
        self.fdb2 = featureDenoiseBlk(64+64+4,64+64+4)
        self.fracPool2 = nn.FractionalMaxPool2d(3, output_size=(16, 16))
        self.encoder2 = EncoderBlock(32, 64+64+4, 4, 2, 16, 4, 660, 0.2)
        
        self.fdb3 = featureDenoiseBlk(64+64+4,64+64+4)
        self.fracPool3 = nn.FractionalMaxPool2d(3, output_size=(8, 8))
        self.encoder3 = EncoderBlock(16, 64+64+4, 4, 2, 16, 4, 660, 0.2)
        
        self.fdb4 = featureDenoiseBlk(64+64+4,64+64+4)
        
        self.upsampl1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsampl2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsampl3 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.upConv1 = nn.Conv2d(in_channels = 64+64+4, out_channels = 64+64+4, kernel_size = 3, dilation = 1, stride = 1, padding = 'same', padding_mode = 'reflect')
        self.upConv2 = nn.Conv2d(in_channels = 64+64+4, out_channels = 64+64+4, kernel_size = 3, dilation = 1, stride = 1, padding = 'same', padding_mode = 'reflect')
        self.upConv3 = nn.Conv2d(in_channels = 64+64+4, out_channels = 64+64+4, kernel_size = 3, dilation = 1, stride = 1, padding = 'same', padding_mode = 'reflect')
        
        self.hAttnConv = nn.Conv2d(in_channels = 64+64+4, out_channels = 64+64+4, kernel_size = 3, dilation = 1, stride = 1, padding = 'same', padding_mode = 'reflect')
        
        self.conv1 = nn.Conv2d(in_channels = 64+64+4, out_channels = 1, kernel_size = 3, dilation = 1, stride = 1, padding = 'same', padding_mode = 'reflect')
        self.BN1 = nn.BatchNorm2d(1)
        self.layernorm1 = nn.LayerNorm(64+64+4)
       
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, x: Tensor,
                      x_mask: Tensor=None, y_mask: Tensor=None) -> Tensor:         
        noisy = x
        sobel_edge = self.sobel_conv(x)
        tmapHF = self.tmapHF(x)
        tmapLF = self.tmapLF(x)
        tmapFM = torch.cat((tmapHF, tmapLF, sobel_edge), dim = 1)        
        outFM = self.reluFM(self.convFM(tmapFM))
        x1 = self.fdb1(tmapFM)
        x2 = self.fracPool1(x1)
        x1_skip = self.encoder1(x1)
        
        x3 = self.fdb2(x2)
        x4 = self.fracPool2(x3)
        x3_skip = self.encoder2(x3)
        
        x5 = self.fdb3(x4)
        x6 = self.fracPool3(x5)
        x5_skip = self.encoder3(x5)
        
        x7 = self.fdb4(x6)
        
        x8 = self.upConv1(self.upsampl1(x7))       
        x9 = torch.add(x8,torch.mul(x5,F.sigmoid(x5_skip)))
        
        x10 = self.upConv2(self.upsampl2(x9))         
        x11 = torch.add(x10,torch.mul(x3,F.sigmoid(x3_skip)))
        
        x12 = self.upConv3(self.upsampl3(x11))        
        x13 = torch.add(x12, torch.mul(x1,F.sigmoid(x1_skip)))
        
        x13 = rearrange(x13,'b c h w -> b h w c')
        x13 = self.layernorm1(x13)
        x13 = rearrange(x13, 'b h w c -> b c h w')
        
        x5_skip = F.interpolate(x5_skip, size = 64, mode = 'bilinear')
        x3_skip = F.interpolate(x3_skip, size = 64, mode = 'bilinear')
        x14 = x5_skip+x3_skip+x1_skip
        x14 = torch.mul(x13,F.sigmoid(self.hAttnConv(x14)))        
        out = self.BN1(self.conv1(x14))        
        return outFM, out
