import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import time
from einops import rearrange
import numbers

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
       
        mu = x.mean(-1, keepdim=True)
      
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class AttentionWithPrompt(nn.Module):
    def __init__(self, dim, num_heads, bias, num_prompt_tokens=5):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_prompt_tokens = num_prompt_tokens

        self.prompt_embed = nn.Parameter(torch.randn(1, dim, num_prompt_tokens))  # (1, C, P)

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)       # Each: (B, C, H, W)

        # 插入 prompt：扩展为 (B, C, H*W + P)
        prompt = self.prompt_embed.expand(B, -1, -1)  # (B, C, P)
        q = torch.cat([prompt, q.flatten(2)], dim=2)  # (B, C, P + HW)
        k = torch.cat([prompt, k.flatten(2)], dim=2)
        v = torch.cat([prompt, v.flatten(2)], dim=2)

        # rearrange to (B, head, C//head, P+HW)
        q = rearrange(q, 'b (head c) n -> b head c n', head=self.num_heads)
        k = rearrange(k, 'b (head c) n -> b head c n', head=self.num_heads)
        v = rearrange(v, 'b (head c) n -> b head c n', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, head, N, N)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, head, C', N)
        out = rearrange(out, 'b head c n -> b (head c) n')  # (B, C, N)

        # 去掉 prompt token（只保留后面的 H*W）
        out = out[:, :, self.num_prompt_tokens:]  # (B, C, H*W)
        out = out.view(B, C, H, W)

        return self.project_out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, prompt=False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if prompt:
            self.attn = AttentionWithPrompt(dim, num_heads, bias)
        else:
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
 
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, 
                 dim, 
                 window_size, 
                 num_heads, 
                 qkv_bias=True, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim)
        
        self.proj_drop = nn.Dropout(proj_drop) 
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        out_vis =  dict()
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        out_vis['inner_prod'] = attn.detach()

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        out_vis['rpb'] = relative_position_bias.unsqueeze(0).detach()
       
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0] # nW, windowsize*windowsize, windowsize*windowsize
            ## Broadcast the mask matrix to the batch dimension
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)  # nw*B, nh, windowsize*windowsize, windowsize*windowsize
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            

        out_vis['attn'] = attn.detach()
      
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, out_vis
        
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, img_N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * img_N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * img_N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += img_N * self.dim * self.dim
        return flops


class WindowAttention_Prompt(WindowAttention):
    def __init__(self, 
                 num_prompts, 
                 dim, 
                 window_size, 
                 num_heads,
                 qkv_bias=True, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,*args, **kwargs):
        super().__init__(dim, 
                         window_size, 
                         num_heads, 
                         qkv_bias, 
                         qk_scale, 
                         attn_drop, 
                         proj_drop, *args, **kwargs)
        self.num_prompts = num_prompts
  
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        out_vis = {}
        B_, N, C = x.shape
        fin_N = N - self.num_prompts
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0][:,:,self.num_prompts:], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        # out_vis['inner_prod'] = attn.detach()

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])

        
        # expand relative_position_bias
        _C, _H, _W = relative_position_bias.shape

        relative_position_bias = torch.cat((
            torch.zeros(_C, _H , self.num_prompts, device=attn.device),
            relative_position_bias
            ), dim=-1)
   

        # print(f"attn shape: {attn.shape}, relative_position_bias shape: {relative_position_bias.shape}")
       
        attn = attn + relative_position_bias.unsqueeze(0)
        out_vis['attn_beforesm'] = attn.detach()

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 64, 64 + n_prompts ) 
            nW, _H, _W = mask.shape
            attn = attn.view(B_ // nW, nW, self.num_heads, fin_N, N) + mask.unsqueeze(1).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, fin_N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        out_vis['attn'] = attn.detach()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, fin_N, C)
        # out_vis['x'] = x.detach()
        x = self.proj(x)
        # out_vis['x_proj'] = x.detach()
        x = self.proj_drop(x)
        return x, out_vis
        
        
class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, 
                 config, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=7, 
                 shift_size=0,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.actual_resolution = None
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.args = config
        
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.num_prompts = 1 
    
        self.attn = WindowAttention(
                    dim, 
                    window_size=to_2tuple(self.window_size), 
                    num_heads=num_heads,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale, 
                    attn_drop=attn_drop, 
                    proj_drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
    
        self.register_buffer("attn_mask", attn_mask)
        
    def calculate_mask(self, x_size, window_size, shift_size, device):
        # calculate attention mask for SW-MSA
        
        H, W = x_size
        H_p = int(np.ceil(H / window_size)) * window_size
        W_p = int(np.ceil(W / window_size)) * window_size
       
        img_mask = torch.zeros((1, H_p, W_p, 1),device=device) # 1 H W 1
        
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    
    
    def forward(self, x, x_size):
        ##  m(1,c)
        ### mutil scale training
        
        self.actual_resolution = x_size
        H, W = self.actual_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size, "

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        H_p = int(np.ceil(H / self.window_size)) * self.window_size
        W_p = int(np.ceil(W / self.window_size)) * self.window_size 
        x_padded = F.pad(x, (0, 0, 0, W_p-W, 0, H_p-H),)
       
      
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_padded, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_padded
        # partition windows
      
        mask = self.calculate_mask(x_size, self.window_size, self.shift_size, x.device)
    
     
     
        # print(f"1:{time2-time1}")
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
     
        x_windows = x_windows
       
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # if self.eval:
        #     mask = self.calculate_mask(x_size, self.window_size, self.shift_size).to(x.device)
            
        attn_windows, attn_values = self.attn(x_windows,  mask=mask)  # nW*B, window_size*window_size, C
       
        #print(f"4:{time5-time4}")
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_p, W_p)  # B H' W' C
       
        # print(f"5:{time6-time5}")
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if H_p-H > 0 or W_p-W > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H * W, C)

 
        # print(f"6:{time7-time6}")
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        attn_values['x'] = x.detach()
    
        # print(f"7:{time8-time7}")
        return x, attn_values
        
class SwinTransformerBlock_Prompt(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, 
                 config, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=7, 
                 shift_size=0,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.actual_resolution = None
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.args = config
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.num_prompts = 1 
       
        self.attn = WindowAttention_Prompt(
                    self.num_prompts,
                    dim, 
                    window_size=to_2tuple(self.window_size),
                    num_heads=num_heads, 
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    attn_drop=attn_drop, 
                    proj_drop=drop)
        
    
    
    def calculate_mask(self, x_size, window_size, shift_size):
        # calculate attention mask for SW-MSA
       
        H, W = x_size
        H_p = int(np.ceil(H / window_size)) * window_size
        W_p = int(np.ceil(W / window_size)) * window_size
        img_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    
    
    def forward(self, x, m, x_size):
        ##  m(1,c)
        ### mutil scale training
        self.actual_resolution = x_size
        H, W = self.actual_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        pad_h = int(np.ceil(H / self.window_size)) * self.window_size
        pad_w = int(np.ceil(W / self.window_size)) * self.window_size 
        x_padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h),)
        _, H_p, W_p, _ = x_padded.shape
        num_windows = (H_p // self.window_size) * (W_p // self.window_size)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_padded, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_padded
        # partition windows
        
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        ####mutio scale training    calculate_mask in forward
        mask_a = self.calculate_mask(x_size, self.window_size, self.shift_size)
     
        
          
        m = m.repeat(B, num_windows, 1)
        m = m.view(B, H // self.window_size, W // self.window_size, C )
        shifted_m = m
        # partition windows
        m_windows = window_partition(shifted_m, 1)  # nW*B, window_size, window_size, C
        
        ##m_windows = m_windows.view(-1,(self.window_size//self.mask_down) * (self.window_size//self.mask_down), C)  # nW*B, window_size*window_size, C
        m_windows = m_windows.view(-1, 1, C)  
        
        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        x_windows = torch.cat((m_windows, x_windows), dim=1)
        
        #mask_b = torch.nn.functional.interpolate(mask_a[:,:,(self.window_size//self.mask_down)**2:self.window_size**2-(self.window_size//self.mask_down)**2].unsqueeze(1),(self.window_size**2,(self.window_size//self.mask_down)**2)).squeeze(1)
        mask_b = torch.nn.functional.interpolate(mask_a[:,:,1:self.window_size**2-1].unsqueeze(1),(self.window_size**2,1)).squeeze(1)
        prompt_mask = torch.cat([mask_a,mask_b],2)
         

       
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, attn_values = self.attn(x_windows, mask=prompt_mask.to(x.device))  # nW*B, window_size*window_size, C
    
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

    
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        attn_values['x'] = x.detach()

        return x, attn_values
    

class RSTB(nn.Module):
    def __init__(self,
                 dim, 
                 input_resolution, 
                 depth, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint=False, 
                 config=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.args = config
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.args = config
        # print("=======================")
        # print(dim)
        self.blocks = nn.ModuleList([
                SwinTransformerBlock(       
                    config,
                    dim=dim, 
                    input_resolution=input_resolution,
                    num_heads=num_heads, 
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop, 
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # noqa
                    norm_layer=norm_layer)
                for i in range(depth)])
        
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()


    def forward(self, x, x_size):
        out = self.patch_embed(x)
        attns = []
        #time1 = time.time()
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                out = checkpoint.checkpoint(blk, out)
            else:
                if self.args.RETURN_ATTENTION:
                
                    out, attn = blk(out,x_size)
                    attns.append(attn)
                else:
                
                    out, _ = blk(out, x_size)
                    attn = None
                    attns.append(attn)
        #time2 = time.time()
        #print(f'rstb:{time2-time1}')
        output = self.patch_unembed(out, x_size) + x
        attns.append(output.detach())
      
        return output, attns

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
   
   
     
class RSTB_Prompt(nn.Module):
    def __init__(self,
                 dim, 
                 input_resolution, 
                 depth, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint=False, 
                 use_prompt=True, 
                 config=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.args = config
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.args = config
        
        self.blocks = nn.ModuleList([
                SwinTransformerBlock_Prompt(
                    use_prompt,
                    config,
                    dim=dim, 
                    input_resolution=input_resolution,
                    num_heads=num_heads, 
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop, 
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # noqa
                    norm_layer=norm_layer)
                for i in range(depth)])
        
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()


    def forward(self, x, prompt, x_size):
        out = self.patch_embed(x)
        attns = []
        
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                out = checkpoint.checkpoint(blk, out)
            else:
                if self.args.RETURN_ATTENTION:
                    out, attn = blk(out, prompt, x_size)
                    attns.append(attn)
                else:
                    out, _ = blk(out, prompt, x_size)
                    attn = None
                    attns.append(attn)
        output = self.patch_unembed(out, x_size) + x
        attns.append(output.detach())
        return output, attns

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"