import math
#from re import X
from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv = DWConv(hidden_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x,x_size):
        x = self.fc1(x)
        x = self.act(x)

        x = self.dwconv(x,x_size)
        x = self.act(x)     #B C H W

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

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
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
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

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x),x_size))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #self.reduction = nn.Linear(4 * dim,  dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        #H, W = self.input_resolution
        H, W = x_size[0], x_size[1]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, (x_size[0] // 2, x_size[1] // 2)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        #self.norm = norm_layer(dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        #H, W = self.input_resolution
        H, W = x_size[0], x_size[1]
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x, (x_size[0] * 2, x_size[1] * 2)




class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x, x_size = self.downsample(x, x_size)
        return x, x_size

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.upsample is not None:
            x , x_size= self.upsample(x, x_size)
        return x, x_size

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=64, embed_dim=192, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

class Dconvbasic(nn.Module):
    def __init__(self, embed_dim = 192):
        super().__init__()
        self.embed_dim = embed_dim
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim),
            nn.PReLU()
        )


    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        x = self.dconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                # m.append(nn.Sequential(nn.Conv2d(num_feat,num_feat,3,1,1),
                #          nn.PReLU(),
                #          nn.Conv2d(num_feat,4 * num_feat,1,1,0)))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            # m.append(nn.Sequential(nn.Conv2d(num_feat,num_feat,3,1,1),
            #              nn.PReLU(),
            #              nn.Conv2d(num_feat,9 * num_feat,1,1,0)))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



class CrossAttention(nn.Module):
    def __init__(self, dim, y_dim, num_heads, bias):                 
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim                      #dim(192) is depth map channels num, y_dim(384) is color map channel num
        self.y_dim = y_dim
        ##self.y_num_heads = y_num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.SpatiaGate = SpatialGate(dim=dim,y_dim=y_dim)

        self.y_qkv = nn.Conv2d(y_dim, y_dim*2, kernel_size=1, bias=bias)
        self.x_qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x_qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.y_qkv_dwconv = nn.Conv2d(y_dim*2, y_dim*2, kernel_size=3, stride=1, padding=1, groups=y_dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, y, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.dim, x_size[0], x_size[1])
        y = y.transpose(1, 2).view(B, self.y_dim, x_size[0], x_size[1])
        b,c,h,w = x.shape
        b_y, c_y, h_y, w_y = y.shape

        q = self.x_qkv_dwconv(self.x_qkv(x))
        kv = self.y_qkv_dwconv(self.y_qkv(y))
        k,v = kv.chunk(2, dim=1)   
        #v = self.SpatiaGate(q,v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c_y) h w -> b head c_y (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c_y) h w -> b head c_y (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = out.flatten(2).transpose(1, 2)
        return out


class CrossTransformerblock(nn.Module):
    def __init__(self, dim, y_dim,input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim                                       #dim(192) is depth map channels num, y_dim(384) is color map channel num
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm1_y = norm_layer(y_dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = CrossAttention(dim, y_dim, num_heads,qkv_bias)

        # self.spatial = Spatia(dim=dim)   #门机制

    def forward(self, x, y, x_size):
        H, W = x_size
        #H_Y, W_Y = y_size
        B, L, C = x.shape
        B_Y, L_Y, C_Y =y.shape
        # assert L == H * W, "input feature has wrong size"
        # x, y = self.spatial(x,y,x_size)
        shortcut = x
        x = self.norm1(x)

        y = self.norm1_y(y)
        x = self.attn(x, y, x_size) + shortcut
        x = x + self.mlp(self.norm2(x),x_size)
        return x



class FusionChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(FusionChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class bi_model_x4(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, args ,img_size=32, img_size_y=128,scale_factor=4,patch_size=1, in_chans=1, out_chans=1,in_chans_y=3,
                 embed_dim=64,de_features=64,embed_dim_y=32,coex_features=32, up_heads=[1,2],num_heads=[4,2],depths=[2,2,2,2], guid_depths=[2,2,2,2], 
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        #print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        #depths_decoder,drop_path_rate,num_classes))


        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.down_num = int(math.log(scale_factor,2))
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        #self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features = int(embed_dim)
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.window_size = window_size
        self.coex_features = coex_features
        self.de_features = de_features
        self.scale_factor = scale_factor
        
        #print(self.window_size)
        #extract fearures
        #self.Fconv_1 = nn.Conv2d(1,64,3,1,1)
        #self.Fconv = nn.Conv2d(3,64,3,1,1)
        self.Fconv_1 = nn.Sequential(nn.Conv2d(1,self.de_features,kernel_size=5,stride=1,padding=2),
                                    # nn.BatchNorm2d(self.de_features),
                                    nn.Conv2d(self.de_features,self.de_features,kernel_size=3,stride=1,padding=1)
                                    )
        self.Fconv = nn.Sequential(nn.Conv2d(3,self.coex_features,kernel_size=5,stride=1,padding=2),
                                    # nn.BatchNorm2d(self.coex_features),
                                    nn.Conv2d(self.coex_features,self.coex_features,kernel_size=3,stride=1,padding=1)
                                    )
        

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=self.de_features, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_embed_y = PatchEmbed(
            img_size=img_size_y, patch_size=patch_size, in_chans=self.coex_features, embed_dim=embed_dim_y,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches_y = self.patch_embed_y.num_patches
        patches_resolution_y = self.patch_embed_y.patches_resolution
        self.patches_resolution_y = patches_resolution_y


        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            self.absolute_pos_embed_y = nn.Parameter(torch.zeros(1, num_patches_y, embed_dim))
            trunc_normal_(self.absolute_pos_embed_y, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_d = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_g = [x.item() for x in torch.linspace(0, drop_path_rate, sum(guid_depths))]
        
        
        #guidance image downsample block
        self.down_depths = []
        for i in range(self.down_num):
            self.down_depths.append(2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.down_depths))]  # stochastic depth decay rule
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.down_num):
            layer = BasicLayer(dim=int(embed_dim_y * 2 ** i_layer),
                               input_resolution=(patches_resolution_y[0] // (2 ** i_layer),
                                                 patches_resolution_y[1] // (2 ** i_layer)),
                               depth=self.down_depths[i_layer],
                            #    num_heads=num_heads[0] * 2 ** i_layer,
                               num_heads=up_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(self.down_depths[:i_layer]):sum(self.down_depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging, #if (i_layer < self.down_num - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.downsample_layers.append(layer)
            

        #guidance image extract deep fearures
        self.guidedlayer0 = BasicLayer(dim=int(embed_dim_y * 2 ** self.down_num),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=guid_depths[0],
                            #    num_heads=num_heads[0] * 2 ** self.down_num,
                               num_heads=num_heads[0],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_g[sum(guid_depths[:0]):sum(guid_depths[:1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.guid_dconv0 = Dconvbasic(embed_dim=int(embed_dim_y * 2 ** self.down_num))
        self.guidedlayer1 = BasicLayer(dim=int(embed_dim_y * 2 ** self.down_num),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=guid_depths[1],
                            #    num_heads=num_heads[0] * 2 ** self.down_num,
                               num_heads=num_heads[0],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_g[sum(guid_depths[:1]):sum(guid_depths[:2])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.guid_dconv1 = Dconvbasic(embed_dim=int(embed_dim_y * 2 ** self.down_num))
        self.guidedlayer2 = BasicLayer(dim=int(embed_dim_y * 2 ** self.down_num),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=guid_depths[2],
                            #    num_heads=num_heads[0] * 2 ** self.down_num,
                               num_heads=num_heads[0],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_g[sum(guid_depths[:2]):sum(guid_depths[:3])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.guid_dconv2 = Dconvbasic(embed_dim=int(embed_dim_y * 2 ** self.down_num))

        
        self.guidedlayer3 = BasicLayer(dim=int(embed_dim_y * 2 ** self.down_num),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=guid_depths[3],
                            #    num_heads=num_heads[0] * 2 ** self.down_num,
                               num_heads=num_heads[0],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_g[sum(guid_depths[:3]):sum(guid_depths[:4])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.guid_dconv3 = Dconvbasic(embed_dim=int(embed_dim_y * 2 ** self.down_num))
        
        
        #CrossAttention
        self.crosslayer_d0 = CrossTransformerblock(dim = int(embed_dim),
                                y_dim= int(embed_dim_y * 2 ** self.down_num),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        self.crosslayer_d1 = CrossTransformerblock(dim = int(embed_dim),
                                y_dim= int(embed_dim_y * 2 ** self.down_num),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        self.crosslayer_d2 = CrossTransformerblock(dim = int(embed_dim),
                                y_dim= int(embed_dim_y * 2 ** self.down_num),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        
        self.crosslayer_c0 = CrossTransformerblock(dim = int(embed_dim_y * 2 ** self.down_num),
                                y_dim= int(embed_dim),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        self.crosslayer_c1 = CrossTransformerblock(dim = int(embed_dim_y * 2 ** self.down_num),
                                y_dim= int(embed_dim),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        self.crosslayer_c2 = CrossTransformerblock(dim = int(embed_dim_y * 2 ** self.down_num),
                                y_dim= int(embed_dim),
                                input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                                num_heads=num_heads[1],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias)
        
        #depth map                       
        self.depthlayer0 = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[0],
                               num_heads=num_heads[1],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_d[sum(depths[:0]):sum(depths[:1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.depth_dconv0 = Dconvbasic(embed_dim=int(embed_dim))
        self.depthlayer1 = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[1],
                               num_heads=num_heads[1],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_d[sum(depths[:1]):sum(depths[:2])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.depth_dconv1 = Dconvbasic(embed_dim=int(embed_dim))
        self.depthlayer2 = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[2],
                               num_heads=num_heads[1],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_d[sum(depths[:2]):sum(depths[:3])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        # self.depth_dconv2 = Dconvbasic(embed_dim=int(embed_dim))
        self.depthlayer3 = BasicLayer(dim=int(embed_dim),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[3],
                               num_heads=num_heads[1],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_d[sum(depths[:3]):sum(depths[:4])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
        
        self.before_upconv = nn.Sequential(nn.Conv2d(in_channels=4 * embed_dim, out_channels=64, kernel_size=1,stride=1,padding=0),
                                            nn.PReLU()
                                            )

                                
        self.upsample = Upsample(self.scale_factor, 64)
        self.output = nn.Conv2d(in_channels=64, out_channels=self.out_chans, kernel_size=3,stride=1,padding=1)

        ###############color branch upsample and output##################
        
        
        self.before_upconv_c = nn.Sequential(nn.Conv2d(in_channels=int(embed_dim_y * 2 ** self.down_num), out_channels=64, kernel_size=1,stride=1,padding=0),
                                            nn.PReLU()
                                            )

        
        self.upsample_c = Upsample(self.scale_factor, 64)
        self.output_c = nn.Conv2d(in_channels=64, out_channels=self.out_chans, kernel_size=3,stride=1,padding=1)

        #Fusion depth
        self.fusionattention = FusionChannelAttention(in_planes=128)
        self.fusionoutput = nn.Conv2d(in_channels=128, out_channels=self.out_chans, kernel_size=3,stride=1,padding=1)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        #mod_h = h % (self.window_size * 4)
        #mod_w = w % (self.window_size * 4)
        #mod_pad_h = ((self.window_size * 4) - mod_h) % (self.window_size * 4)
        #mod_pad_w = ((self.window_size * 4) - mod_w) % (self.window_size * 4)
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        #x = torch.cat([x, torch.flip(x,[2])], 2)[]
        return x
    def forward_down(self, y, y_size):
        
        for layer in self.downsample_layers:
            y, y_size = layer(y, y_size)
        return y, y_size

    def forward(self, x, y):                                          #  x is depth, y is color
        #if x.size()[1] == 1:
            #x = x.repeat(1,3,1,1)
        H, W = y.shape[2], y.shape[3]
        hx, wx = x.shape[2], x.shape[3]
        
        # x_lr = x
        x = self.check_image_size(x)
        y = self.check_image_size(y)
        x_size  = (x.shape[2], x.shape[3])
        y_size  = (y.shape[2], y.shape[3])
        x = x
        y = y
        res = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        x = self.Fconv_1(x)
        y = self.Fconv(y)
        x = self.patch_embed(x)
        y = self.patch_embed_y(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed_y
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        y, y_size = self.forward_down(y, y_size)

        y1, y_size = self.guidedlayer0(y, y_size)
        

        


        x1, x_size = self.depthlayer0(x, x_size)
       

        x2 = self.crosslayer_d0(x1, y1, x_size)
        x2, x_size = self.depthlayer1(x2, x_size)
       

        y2 = self.crosslayer_c0(y1, x2, x_size)
        y2, y_size = self.guidedlayer1(y2,y_size)
        

        x3 = self.crosslayer_d1(x2, y2, x_size)
        x3, x_size = self.depthlayer2(x3, x_size)
        

        y3 = self.crosslayer_c1(y2, x3, x_size)
        y3, y_size = self.guidedlayer2(y3,y_size)
        

        x4 = self.crosslayer_d2(x3, y3, x_size)
        x4, x_size = self.depthlayer3(x4, x_size)
       

        y4 = self.crosslayer_c2(y3, x4, x_size)
        y4, y_size = self.guidedlayer3(y4,y_size)
       
        x = torch.cat([x1, x2, x3, x4],2)
        
        B, L, C = x.shape
        x = x.view(B, hx, wx, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        
        x = self.before_upconv(x)
        x = self.upsample(x)
        

        # B_y, L, C = y4.shape
        y = y4.view(B, hx, wx, -1)
        y = y.permute(0, 3, 1, 2)  # B,C,H,W
        y = self.before_upconv_c(y)
        y = self.upsample_c(y)
        

        z = torch.cat([x,y],1)
        z = self.fusionattention(z) * z
        

        x = self.output(x)
        y = self.output_c(y)
        z = self.fusionoutput(z)
        
        return x + res, y + res, z + res
       
def make_model(args, parent=False):
    return bi_model_x4(args)