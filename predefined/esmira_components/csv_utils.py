import torch
import torch.nn as nn
from einops import rearrange
from typing import Union, Literal
'''
V2 -- pw*pl*c
V1 -- c
'''

def conv_nxn_bn_group(inp:int, oup:int, kernal_size:Union[int, tuple]=3, stride:Union[int, tuple]=1, groups:int=4):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim:int, fn:nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, dropout:float=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim:int, heads:int=8, dim_head:int=64, dropout:float=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # B, (ph*pw), L/ph*W/pw, C
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # B, (ph*pw), L/ph*W/pw, C --> B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads)*3
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        # B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads) --> B, (ph*pw), heads, L/ph*W/pw, dim_head

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] matmul --> [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        # [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw] X [B, (ph*pw), heads, L/ph*W/pw, dim_head] -->
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head]
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] --> [B, (ph*pw), L/ph*W/pw, dim_head*heads]
        return self.to_out(out)  # [B, (ph*pw), L/ph*W/pw, dim_head*heads] --> [B, (ph*pw), L/ph*W/pw, C]


class AttentionV2(nn.Module):
    def __init__(self, dim:int, heads:int=8, dim_head:int=64, dropout:float=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # B, (ph*pw), L/ph*W/pw, C
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # B, (ph*pw), L/ph*W/pw, C --> B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads)*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), qkv)
        # B, (ph*pw), L/ph*W/pw, inner_dim(dim_head*heads) --> B, (ph*pw), heads, L/ph*W/pw, dim_head

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] matmul --> [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        # [B, (ph*pw), heads, L/ph*W/pw, L/ph*W/pw] X [B, (ph*pw), heads, L/ph*W/pw, dim_head] -->
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head]
        out = rearrange(out, 'b n h d -> b n (h d)')
        # [B, (ph*pw), heads, L/ph*W/pw, dim_head] --> [B, (ph*pw), L/ph*W/pw, dim_head*heads]
        return self.to_out(out)  # [B, (ph*pw), L/ph*W/pw, dim_head*heads] --> [B, (ph*pw), L/ph*W/pw, C]


class ParrellelAttention(nn.Module):
    def __init__(self, dim:int, heads:int=8, dim_head:int=64, dropout:float=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        v = x.clone()
        qk = self.to_qkv(x).chunk(2, dim=-1)
        q, k= map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qk)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class ParrellelAttentionV2(nn.Module):
    def __init__(self, dim:int, heads:int=8, dim_head:int=64, dropout:float=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        v = x.clone()
        qk = self.to_qkv(x).chunk(2, dim=-1)
        q, k= map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), qk)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim:int, depth:int, heads:int, dim_head:int, mlp_dim:int, dropout:float=0.,
                 attn_type:str='normal'):
        super().__init__()
        self.attn_type = attn_type
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if attn_type=='mobile':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
                ]))
            elif attn_type=='normal':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, AttentionV2(dim, heads, dim_head, dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
                ]))
            elif attn_type=='parr_mobile':
                self.layers.append(nn.ModuleList([
                    ParrellelAttention(dim, heads, dim_head, dropout),
                    FeedForward(dim, mlp_dim, dropout)
                ]))
            elif attn_type=='parr_normal':
                self.layers.append(nn.ModuleList([
                    ParrellelAttentionV2(dim, heads, dim_head, dropout),
                    FeedForward(dim, mlp_dim, dropout)
                ]))
    
    def forward(self, x):
        # B, (ph*pw), L/ph*W/pw, C
        for attn, ff in self.layers:
            if 'parr' in self.attn_type:
                x = attn(x) + ff(x) # + x # not in the original similifying tranformer blocks
            else:
                x = attn(x) + x
                x = ff(x) + x
        return x
            

class ViTBlock(nn.Module):
    def __init__(self, channel:int, kernel_size:Union[int, tuple], patch_size:Union[int, tuple], 
                 groups:int, depth:int, mlp_dim:int, dropout:float=0., 
                 attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='normal',
                 out_channel:Union[int, None]=None):
        super().__init__()
        '''
        ViTBlock: simplified ViT block, merging channel and dim
        channel:(channels of input), 
        depth:(num of transformer block)[2,4,3],
        kernel_size:(kernel size of convlutional neural networks)
        patch_size:(patch size of transformer)
        heads:(heads number/kernel number)
        att_dim:(nodes of mlp in attention module)
        mlp_dim:(nodes of mlp in feedfward module)
        groups:(groups for convolution)
        dropout
        '''
        out_ch = out_channel if out_channel is not None else channel
        self.attn_type = attn_type
        if type(patch_size)==list or type(patch_size)==tuple:
            self.ph, self.pw = patch_size  
        else:
            self.ph, self.pw = patch_size, patch_size
        tfdim = channel*self.ph*self.pw if 'normal' in attn_type else channel
        self.transformer = Transformer(tfdim, depth, 4, 8, mlp_dim, dropout, attn_type=attn_type)
        # Transformer(dim(channels of input), depth(num of transformer block)[2,4,3], 
        #             4(heads number/kernel number), 8(length of mlp in attention),
        #             mlp_dim(nodes of mlp, extension), dropout)
        self.merge_conv = conv_nxn_bn_group(2 * channel, out_ch, kernel_size, stride=1, groups=groups)
    
    def forward(self, x):
        # input size: B, 4*C, L, W / B, C, D, L, W not included
        y = x.clone()
        
        # Global representations
        _, _, h, w = x.shape
        if 'mobile' in self.attn_type:
            x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
            # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
            x = self.transformer(x)
            x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        else:
            x = rearrange(x, 'b d (h ph) (w pw) -> b  (h w) (ph pw d)', ph=self.ph, pw=self.pw)
            # B, C, L, W --> B, (ph*pw), L/ph*W/pw, C
            x = self.transformer(x)
            x = rearrange(x, 'b (h w) (ph pw d) -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # x: B, 4*C, L, W 

        # Fusion
        x = torch.cat((x, y), 1)
        # x: B, 4*2C, L, W
        x = self.merge_conv(x)
        # x: B, 4*C, L, W
        return x