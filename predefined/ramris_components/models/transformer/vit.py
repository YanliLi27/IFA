import torch
from torch import nn
from models.utils.attn_sup import cls_repeat, rearrange, WindowCreator

# Vision Transformer的核心组件：
# (1) Patch embedding模块：从图片[b, c, l, w] 变成 [b, (l/pre, w/pre), (pre*pre*c)]再变成 [b, (l/pre, w/pre), dim]
# (2) cls token: 长度为n, 用于加载到输入上进行分类操作 [b, (l/pre, w/pre), dim] --> [b, (l/pre, w/pre)+n, dim]
# (3) Positional embedding: 提供位置信息
# (4) Transformer的主轴：其超参数如下：
# dim, depth, heads, dim_head, mlp_dim, dropout
# dim是（输入的长度） 例如一个batch*3*512*512的图像，传入后，采用32*32作为一个词，变成batch*(16*16)*（3*32*32）
# 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)' 
# 其中的（3*32*32）/(p1 p2 c) 就是原始长度，进行词向量化后从该长度转化为dim长度
# depth是encoder block的数量
# heads是多头自注意力的头数，即CNN的kernel数
# dim是词向量长度
# dim_head是（经过转换的词向量在注意力头的预设压缩长度）
# mlp_dim是全连接层FFN的隐藏层节点数量
# 输入为[b, (l/pre, w/pre)+n, dim]， 输出为[b, (l/pre, w/pre)+n, dim]
# (5) CLS分类：[b, (l/pre, w/pre)+n, dim]的[b, :n, dim]取出，并放入一个[dim, classes]中进行分类


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # x [b, 1+cmm, dim]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # x [b, 1+cmm, dim]
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim = -1)
        # [b, 1+cmm, dim] -> [b, 1+cmm, 3* dim_heads * heads] -> [3(for qkv), b, 1+cmm, dim_heads * heads] 
        q, k, v = map(lambda t: rearrange(t, mode='getheads', heads=self.heads), qkv)  # map to take the qkv as an iterator (3, :)
        # [3(for qkv), b, 1+cmm, dim_heads * heads] -> [3(for qkv), b, heads, 1+cmm, dim_heads] 
        # q, k, v = 3(for qkv) * [b, heads, 1+cmm, dim_heads] 

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # ([b, heads, 1+cmm, dim_heads] · [b, heads, dim_heads, 1+cmm])*1/(dim_heads)**0.5
        # --> [b, heads, 1+cmm, 1+cmm]

        attn = self.attend(dots)  # softmax ([b, heads, 1+cmm, 1+cmm])  -- the importance of position in 1+cmm
        attn = self.dropout(attn)  # dropout

        out = torch.matmul(attn, v)  # [b, heads, 1+cmm, 1+cmm] · [b, heads, 1+cmm, dim_heads] --> [b, heads, 1+cmm, dim_heads]
        out = rearrange(out, mode='mergeheads')  # [b, heads, 1+cmm, dim_heads] -> [b, 1+cmm, dim_heads*heads]
        return self.to_out(out) + x  # res block -> [b, 1+cmm, dim_heads*heads]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            WindowCreator(dim_mode=2, p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [b, c, l, w] -> [b, c*m*m, (l/m*w/m] -> [b, 1+cmm, dim]
        b, n, _ = x.shape  # b, c*m*m

        cls_tokens = cls_repeat(self.cls_token, 0, batch_size=b)  # [1, 1, len_word_vec] -> [b, 1, len_word_vec]
        x = torch.cat((cls_tokens, x), dim=1)  # [b, cmm, dim] -> [b, 1+cmm, dim]
        x += self.pos_embedding[:, :(n + 1)]  # [b, 1+cmm, dim] pixel-pixel+value
        x = self.dropout(x)  # [b, 1+cmm, dim] drop some of the value

        x = self.transformer(x)  # [b, 1+cmm, dim] --> [b, 1+cmm, dim]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 分类时采用cls，故选取0位 #[b, 1+cmm, dim] -> [b, 1, dim]

        x = self.to_latent(x)  # no change
        return self.mlp_head(x)  # [b, 1, dim]  -> [b, 1]


class ViT3D(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            WindowCreator(dim_mode=3, p1=patch_height, p2=patch_width, p3=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = cls_repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



if __name__ == '__main__':
    v = ViT(
    image_size = 256,  # 256* 256 * 3
    patch_size = 32,  # 32* 32
    num_classes = 1000,  # 1000类
    dim = 1024,  # 词向量的预设长度 --> 32*32的块 embedding成dim长的向量
    depth = 6,  # 6个encoder
    heads = 16,  # 16kernel的transformer
    mlp_dim = 2048,  # 全连接层节点数
    dropout = 0.1,  # dropout比例
    emb_dropout = 0.1  # 用于mae等操作的mask
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)

    # v = ViT3D(
    # image_size = 128,          # image size
    # frames = 16,               # number of frames
    # image_patch_size = 16,     # image patch size
    # frame_patch_size = 2,      # frame patch size
    # num_classes = 1000,
    # dim = 1024,
    # depth = 6,
    # heads = 8,
    # mlp_dim = 2048,
    # dropout = 0.1,
    # emb_dropout = 0.1
    # )

    # video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

    # preds = v(video) # (4, 1000)