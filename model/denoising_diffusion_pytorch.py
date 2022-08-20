import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

import timm
from einops import rearrange
from .laplacian import LapLoss
from .utils import extract_pre_feat, print0, exists, exists_add, \
                   default, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, unnorm_save_image, \
                   SingletonDataset, fast_randn, fast_randn_like, dclamp, l2norm, fetch_attr 
import os
import copy
from tqdm.auto import tqdm
from collections import namedtuple
from random import random

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


timm_model2dim = { 'resnet34': 512,
                   'resnet18': 512,
                   'repvgg_b0': 1280,
                   'mobilenetv2_120d': 1280,
                   'vit_base_patch8_224': 768,
                   'vit_tiny_patch16_224': 192 }

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        # As residual connections are used in downs features of the UNet, 
        # dividing by 2 keeps the std of these output features similar to ups features in the UNet.
        return 0.5 * ( self.fn(x, *args, **kwargs) + x )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class OutConv(nn.Module):
    # dim_in:   dim of input features x
    # init_dim: dim of residual features from init_conv
    def __init__(self, block_klass, dim_in, init_dim, out_dim):
        super().__init__()
        self.final_res_block = block_klass(dim_in + init_dim, dim_in)
        self.final_conv = nn.Conv2d(dim_in, out_dim, 1)

    # r: residual features from init_conv
    # t: time embedding.
    def forward(self, x, r, t):
        # OutConv may appear in earlier layers of the decoder, in which case x is smaller than r.
        # So resize r to match x.
        r = F.interpolate(r, x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        return x

# https://github.com/lucidrains/denoising-diffusion-pytorch/commit/eba44498d1b33749df6e183bdc0899cf2918a5f3
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules. Kernel size is usually 3.
# No downsampling is done in Block, i.e., it outputs a tensor of the same (H,W) as the input.
class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups = 8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Conv2d(dim, dim_out, kernel_size, padding = padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# Different ResnetBlock's have different mlp (time embedding module).
# The mlp here transforms the input time embedding again to adapt to this block.
# No downsampling is done in ResnetBlock.
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size=3, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),      # Sigmoid Linear Unit, aka swish. https://pytorch.org/docs/stable/_images/SiLU.png
            nn.Linear(time_emb_dim, dim_out * 2),
            nn.LayerNorm(dim_out * 2)       # Newly added.
        ) if exists(time_emb_dim) else None

        # block1 and block2 are ended with a group norm and a SiLU activation.
        self.block1 = Block(dim,     dim_out, kernel_size = kernel_size, groups = groups)
        self.block2 = Block(dim_out, dim_out, kernel_size = kernel_size, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # time embedding is incorporated between block1 and block2.
    def forward(self, x, time_emb = None):
        h = self.block1(x)

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            try:
                time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            except:
                breakpoint()
            scale_shift = time_emb.chunk(2, dim = 1)

        # h has been affine-transformed by scale_shift in block1, 
        # and scale_shift is derived from time_emb.
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

# LinearAttention doesn't compute Q*K similarity (with softmax). 
# Instead, it computes K*V as "context" (after K is softmax-ed).
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, memory_size=0):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.memory_size = memory_size
        if self.memory_size > 0:
            # Persistent memory that serves as a codebook used for image generation.
            self.memory = nn.Parameter(fast_randn(1, dim, 1, memory_size))

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, 1, h * w)
        if self.memory_size > 0:
            x_ext = torch.cat((x_flat, self.memory.expand(b, -1, -1, -1)), dim = 3)
        else:
            x_ext = x_flat

        qkv = self.to_qkv(x_ext).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # q, k are softmax-ed, to ensure that q.k' is still row-normalized (sum to 1).
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        # rescale values to prevent linear attention from overflowing in fp16 setting.
        v = v / (h * w)

        # context: c*c, it's like c centroids, each of c-dim. 
        # q is the weights ("soft membership") of each point belonging to each centroid. 
        # Therefore, it's called "context" instead of "similarity". 
        # Intuitively, it's more like doing clustering on V.
        # https://arxiv.org/abs/1812.01243 Efficient Attention: Attention with Linear Complexities.
        # In contrast, sim in Attention is N*N, with softmax.
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # Remove memory cells.
        out = out[:, :, :, :h*w]
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# Ordinary attention, without softmax.
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 16, memory_size=0):
        super().__init__()
        self.scale = scale # dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.memory_size = memory_size
        if self.memory_size > 0:
            # Persistent memory that serves as a codebook used for image generation.
            self.memory = nn.Parameter(fast_randn(1, dim, 1, memory_size))

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, 1, h * w)
        if self.memory_size > 0:
            x_ext = torch.cat((x_flat, self.memory.expand(b, -1, -1, -1)), dim = 3)
        else:
            x_ext = x_flat    

        qkv = self.to_qkv(x_ext).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # Remove memory cells.
        out = out[:, :, :h*w]

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# denoising model
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        memory_size = 1024,
        num_classes = -1,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16,        
        featnet_type = 'vit_tiny_patch16_224',
        cls_sem_featnet_type = 'vit_tiny_patch16_224',
        finetune_tea_feat_ext = False,
        cls_sem_shares_tea_feat_ext = False,
        distillation_type = 'none',
    ):
        super().__init__()

        # number of input channels
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        # dim = 64, init_dim -> 64.
        init_dim = default(init_dim, dim)
        # image size is the same after init_conv, as default stride=1.
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # init_conv + num_resolutions (4) layers.
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 4 pairs of layers in the encoder/decoder.
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # time embeddings
        # Fixed sinosudal embedding, transformed by two linear layers.
        time_dim = dim * 4              # 256

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, time_emb_dim = time_dim)

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),        # Newly added
        )

        self.num_classes    = num_classes
        # It's possible that self.num_classes < 0, i.e., the number of classes is not provided.
        # In this case, we set cls_embed_type to 'none'.
        if self.num_classes <= 0:
            self.cls_embed_type = 'none'
        else:
            self.cls_embed_type = 'tea_stu'

        if self.cls_embed_type != 'none':
            self.cls_embedding = nn.Embedding(self.num_classes, time_dim)
            self.cls_embed_ln  = nn.LayerNorm(time_dim)
            print0("cls_embedding: ", list(self.cls_embedding.weight.shape))
        else:
            self.cls_embedding = None
            self.cls_embed_ln = None
        # layers

        self.downs  = nn.ModuleList([])
        self.ups    = nn.ModuleList([])
        # ups_tea: teacher decoder
        self.ups_tea = nn.ModuleList([])


        # LinearAttention layers are used in encoder/decoder.
        # Using vanilla attention in encoder/decoder takes 13x RAM, and is 3x slower.

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                # A block_klass is a two-layer conv with residual connection. 
                # block_klass doesn't downsample features.
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                # att(norm(x)) + x.
                # Seems adding memory to encoder hurts performance: image features poluted by memory?
                Residual(PreNorm(dim_in, LinearAttention(dim_in, memory_size=0))),
                # downsampling is done with a 4x4 kernel, stride-2 conv.
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        # Seems the memory in mid_attn doesn't make much difference.
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim, memory_size=0)))
        # Seems setting kernel size to 1 leads to slightly worse images?
        self.mid_block2 = block_klass(mid_dim, mid_dim, kernel_size=1)

        self.featnet_type      = featnet_type
        self.finetune_tea_feat_ext      = finetune_tea_feat_ext
        self.cls_sem_shares_tea_feat_ext   = cls_sem_shares_tea_feat_ext
        self.distillation_type = distillation_type

        if self.featnet_type != 'none' and self.featnet_type != 'mini':
            # Tried 'efficientnet_b0' and 'repvgg_b0', but they didn't perform well.
            self.featnet_type   = self.featnet_type 
            self.featnet_dim    = timm_model2dim[self.featnet_type]

            if self.distillation_type != 'none':
                self.dist_feat_ext_tea  = timm.create_model(self.featnet_type, pretrained=True)
                self.dist_feat_ext_stu  = timm.create_model(self.featnet_type, pretrained=True)
            else:
                self.dist_feat_ext_tea  = None
                self.dist_feat_ext_stu  = None
        else:
            self.featnet_dim  = 0
            self.dist_feat_ext_tea = None
            self.dist_feat_ext_stu = None

        self.cls_sem_featnet_type = cls_sem_featnet_type
        if self.cls_sem_featnet_type != 'none':
            if self.cls_sem_shares_tea_feat_ext:      # default False.
                self.cls_sem_feat_ext = copy.deepcopy(self.dist_feat_ext_tea) 
            else:
                self.cls_sem_feat_ext = timm.create_model(self.cls_sem_featnet_type, pretrained=True)
        else:
            self.cls_sem_feat_ext = None
        
        # distillation_type: 'none' or 'tfrac'. 
        if self.distillation_type == 'none':
            extra_up_dim0 = 0
        # tfrac: teacher sees less-noisy images (either sees a miniature or through a feature network)
        else:
            if self.featnet_type == 'mini':
                extra_up_dim0 = 3
            else:
                # number of output features from the image feature extractor.
                # Both student and teacher have these extra features. 
                # But the features for student are extracted from the original noise images.
                # And the features for teacher are extracted from less noisy images.
                extra_up_dim0 = self.featnet_dim
        
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
                        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            extra_up_dim = extra_up_dim0 if ind == 0 else 0

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in + extra_up_dim, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, memory_size=memory_size))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1),
                # Each block has an OutConv layer, to output multiscale denoised images.
                OutConv(block_klass, dim_in, init_dim, self.out_dim)
            ]))
    
        # Sharing ups seems to perform worse.
        self.tea_share_ups = False
        if self.tea_share_ups:
            self.ups_tea = self.ups
        else:
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)
                extra_up_dim = extra_up_dim0 if ind == 0 else 0
                # miniature image / image features as teacher's priviliged information.

                self.ups_tea.append(nn.ModuleList([
                    block_klass(dim_out + dim_in + extra_up_dim, dim_out),
                    block_klass(dim_out + dim_in, dim_out),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out, memory_size=memory_size))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1),
                    # Each block has an OutConv layer, to output multiscale denoised images.
                    OutConv(block_klass, dim_in, init_dim, self.out_dim)
                ]))

    def pre_update(self):
        if not self.cls_sem_shares_tea_feat_ext:
            # Back up the weight of cls_sem_feat_ext.
            self.cls_sem_feat_ext_backup = copy.deepcopy(self.cls_sem_feat_ext)
        # Otherwise post_update() will restore cls_sem_feat_ext from dist_feat_ext_tea.
        # No need to back up cls_sem_feat_ext.

    def post_update(self):
        if self.cls_sem_shares_tea_feat_ext:
            # Restore cls_sem_feat_ext from teacher feature extractor 
            # to keep it from being updated by the consistency loss.
            self.cls_sem_feat_ext = copy.deepcopy(self.dist_feat_ext_tea)
        else:
            # Restore cls_sem_feat_ext from the backup.
            self.cls_sem_feat_ext = self.cls_sem_feat_ext_backup
            self.cls_sem_feat_ext_backup = None

    def forward(self, x, time, classes_or_embed=None, x_self_cond = None, img_tea=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        init_noise = x
        x = self.init_conv(x)
        r = x.clone() 
        # t: time embedding.
        t = self.time_mlp(time.flatten())

        if self.cls_embed_type != 'none':
            if exists(classes_or_embed):
                # classes_or_embed is 1D, i.e., it contains class labels.
                if classes_or_embed.numel() == classes_or_embed.shape[0]:
                    cls_embed = self.cls_embedding(classes_or_embed)
                # classes_or_embed is already class embeddings.
                else:
                    cls_embed = classes_or_embed

            cls_embed = self.cls_embed_ln(cls_embed)
            # cls_embed: [batch, 1, 1, time_dim=256].
            # cls_embed = cls_embed.view(cls_embed.shape[0], *((1,) * (len(x.shape) - 2)), -1)
            # The teacher always sees the class embedding.
            t_tea = exists_add(t, cls_embed)
            # 'tea_stu': Both the student and teacher see the class embedding.
            # Otherwise, only the teacher sees the class embedding. 
            # But in actual use, we disable these alternative configurations (only 'tea_stu' or 'none').
            t_stu = t_tea if self.cls_embed_type == 'tea_stu' else t
        else:
            t_stu = t_tea = t

        h = []

        for block1, block2, attn, downsample in self.downs:
            # the encoder only sees the time embedding, not the class embedding.
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # All blocks above don't change feature resolution. Only downsample explicitly here.
            # downsample() is a 4x4 conv, stride-2 conv.
            x = downsample(x)

        # x: [16, 512, 16, 16], original image: 128x128, 1/8 of the original image.
        # The last layer of the encoder doesn't have a downsampling layer, but has a dummy Identity() layer.
        # So the original image is downsized by a factor of 2^3 = 8.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        # mid_attn followed by mid_block2 (ResnetBlock), is equivalent to a full transformer layer
        # mid_block2 takes the role of MLP in the transformer.
        x = self.mid_block2(x, t)

        mid_feat = x

        if self.distillation_type == 'tfrac':
        # Always fine-tune the student feature extractor.
            noise_feat = extract_pre_feat(self.featnet_type, self.dist_feat_ext_stu, init_noise, 
                                          mid_feat.shape[2:], 
                                          has_grad=True, use_head_feat=False)
        else:
            noise_feat = None

        preds_stu = []
        for ind, (block1, block2, attn, upsample, out_conv) in enumerate(self.ups):
            if ind == 0 and exists(noise_feat):
                x = torch.cat((x, h[-(2*ind+1)], noise_feat), dim=1)
            else:
                x = torch.cat((x, h[-(2*ind+1)]), dim=1)

            # The decoder sees the sum of the time embedding and class embedding.
            x = block1(x, t_stu)
            x = torch.cat((x, h[-(2*ind+2)]), dim=1)
            x = block2(x, t_stu)
            x = attn(x)
            # All blocks above don't change feature resolution. Only upsample explicitly here.
            # upsample() is a 4x4 conv, stride-2 transposed conv.
            # [64, 256, 16, 16] -> [64, 128, 32, 32] -> [64, 64, 64, 64].
            # upsample() doesn't change the number of channels. 
            # The channel numbers are doubled by block1().
            x = upsample(x)
            pred_stu = out_conv(x, r, t_stu)
            preds_stu.append(pred_stu)

        # img_tea is provided. Do distillation.
        if self.distillation_type == 'tfrac' and exists(img_tea):
            # finetune_tea_feat_ext controls whether to fine-tune the teacher feature extractor.
            # tea_feat is used as the input features to the teacher u-net. Therfore it doesn't matter
            # whether it is in the computation graph.
            tea_feat = extract_pre_feat(self.featnet_type, self.dist_feat_ext_tea, img_tea, 
                                        mid_feat.shape[2:], 
                                        has_grad=self.finetune_tea_feat_ext, use_head_feat=False)
        else:
            tea_feat = None

        if self.distillation_type != 'none' and exists(tea_feat):
            preds_tea = []
            x = mid_feat
            for ind, (block1, block2, attn, upsample, out_conv) in enumerate(self.ups_tea):
                if ind == 0 and exists(tea_feat):
                    x = torch.cat((x, h[-(2*ind+1)], tea_feat), dim=1)
                else:
                    x = torch.cat((x, h[-(2*ind+1)]), dim=1)

                x = block1(x, t_tea)
                x = torch.cat((x, h[-(2*ind+2)]), dim=1)
                x = block2(x, t_tea)
                x = attn(x)
                # All blocks above don't change feature resolution. Only upsample explicitly here.
                # upsample() is a 4x4 conv, stride-2 transposed conv.
                # [64, 256, 16, 16] -> [64, 128, 32, 32] -> [64, 64, 64, 64].
                # upsample() doesn't change the number of channels. 
                # The channel numbers are doubled by block1().                
                x = upsample(x)
                pred_tea = out_conv(x, r, t_tea)
                preds_tea.append(pred_tea)

        else:
            preds_tea = None

        return { 'preds_stu': preds_stu, 'preds_tea': preds_tea, 'noise_feat': noise_feat, 'tea_feat': tea_feat }

# gaussian diffusion trainer class
# suppose t is a 1-d tensor of indices.
def extract_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # out: [b, 1, 1, 1]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: fast_randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: fast_randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# When t is small (~0.5*T), cosb is easier than powa (e=1). When t is large (0.5T-T), powa (e=1) is easier than cosb.
def cosine_beta_schedule(num_timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in "Improved Denoising Diffusion Probabilistic Models".
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# powa (e=1) converges faster, but is too easy for the model to learn.
# Using powa (e=1), a converged model may map noises to bad images.
def power_alpha_schedule(num_timesteps, powa_exponent=3.):
    steps = num_timesteps + 1
    # Avoid setting minimal alphas_cumprod to 0. 
    # Otherwise later sqrt_recip_alphas_cumprod will be 1000, causing NaNs.
    # Setting minimal alphas_cumprod to 0.0064**2, then sqrt_recip_alphas_cumprod[-1] = 156.25.
    base_alphas_cumprod = torch.linspace(1, 0.0064**(2/powa_exponent), steps, dtype = torch.float64)
    alphas_cumprod = base_alphas_cumprod ** powa_exponent
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# Original schedule used in https://github.com/hojonathanho/diffusion
# linb is always harder than cosb. When t is large, the noisy images are too hard to learn.
# For example, the first iteration of denoising result is often bad.
def linear_beta_schedule(num_timesteps):
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype = torch.float64)

def powa_linb_schedule(num_timesteps, powa_exponent=3.):
    powa_betas = power_alpha_schedule(num_timesteps // 2, powa_exponent)
    linb_betas = linear_beta_schedule(num_timesteps // 2)
    powa_linb_betas = torch.cat((powa_betas, linb_betas[1:])).sort()[0]
    powa_linb_betas = powa_linb_betas.unique()
    return powa_linb_betas

def alphas_from_betas(betas):
    alphas = 1. - betas
    # alphas_cumprod: \bar{alpha}_t
    # If using power alpha (e=1) schedule, alphas_cumprod[-1] = 4.1E-5.
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_timesteps = 1000,
        sampling_timesteps = None,
        alpha_beta_schedule = 'cosb',
        powa_exponent = 3.,
        loss_type = 'l1',
        consist_loss_type = 'cosine',
        objective = 'pred_noise',
        featnet_type = 'none',
        distillation_type = 'none',
        distill_t_frac = 0.,
        dataset = None,
        denoise1_cls_sem_loss_use_head_feat = False,
        denoise1_cls_sem_loss_type = 'none',
        denoise1_cls_sem_loss_weight = 0.01,
        align_tea_stu_feat_weight = 0,
        sample_dir = 'samples',      
        ddim_sampling_eta = 1.,
        debug = False,
    ):
        super().__init__()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_master = (self.local_rank <= 0)

        self.num_classes    = fetch_attr(denoise_fn, 'num_classes')
        self.channels       = fetch_attr(denoise_fn, 'channels')
        self.self_condition = fetch_attr(denoise_fn, 'self_condition')

        self.image_size         = image_size
        self.denoise_fn         = denoise_fn    # Unet
        self.objective          = objective
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) ' \
                'or pred_x0 (predict image start)'

        self.featnet_type       = featnet_type
        self.distillation_type  = distillation_type
        self.distill_t_frac     = distill_t_frac if (self.distillation_type == 'tfrac') else -1
        self.dataset            = dataset
        # It's possible that self.num_classes < 0, i.e., the number of classes is not provided.
        # In this case, we set cls_embed_type to 'none'.
        if self.num_classes <= 0:
            self.cls_embed_type = 'none'
        else:
            self.cls_embed_type = 'tea_stu'

        # Use backbone head features for consistency losses, i.e., with the geometric dimensions collapsed.
        # such as class semantics loss (single or interpolation loss, interpolation doesn't work well, though).
        self.denoise1_cls_sem_loss_use_head_feat    = denoise1_cls_sem_loss_use_head_feat
        self.denoise1_cls_sem_loss_type = denoise1_cls_sem_loss_type
        self.denoise1_cls_sem_loss_weight           = denoise1_cls_sem_loss_weight
        self.align_tea_stu_feat_weight  = align_tea_stu_feat_weight
        self.sample_dir = sample_dir
        self.debug = debug
        self.num_timesteps       = num_timesteps
        self.alpha_beta_schedule = alpha_beta_schedule
        self.powa_exponent       = powa_exponent
        self.iter_count = 0

        if self.alpha_beta_schedule == 'cosb':
            print0("Use cosine_beta_schedule")
            betas = cosine_beta_schedule(self.num_timesteps)
        elif self.alpha_beta_schedule == 'powa':
            print0(f"Use power_alpha_schedule (e={self.powa_exponent})")
            betas = power_alpha_schedule(self.num_timesteps, self.powa_exponent)
        elif self.alpha_beta_schedule == 'linb':
            print0("Use linear_beta_schedule")
            betas = linear_beta_schedule(self.num_timesteps)            
        else:
            breakpoint()

        alphas, alphas_cumprod = alphas_from_betas(betas)
        # alphas_cumprod_prev: \bar{alpha}_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.loss_type = loss_type
        self.consist_loss_type = consist_loss_type
        self.laploss_fun    = LapLoss()

        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, num_timesteps) 

        assert self.sampling_timesteps <= num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # sqrt_one_minus_alphas_cumprod -> 1, as t -> T.
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # sqrt_recip_alphas_cumprod -> a big number, as t -> T.
        # If using power alpha (e=1) schedule & T = 1000, torch.sqrt(1. / alphas_cumprod)[-1] = 156.25.
        register_buffer('sqrt_recip_alphas_cumprod',   torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance: \tilde{beta}_t 
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        # posterior_mean_coef1: the coefficients of x_0 in the posterior mean
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # posterior_mean_coef2: the coefficients of x_t in the posterior mean
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # Inference pipeline:                 predict_start_from_noise           q_posterior
    #                    Unet -> noise           ---------->        x_0        ------->    mu_t-1 (mean of x_t-1)
    #                            + x_t
    # Training pipeline:                  predict_start_from_noise
    #                    Unet -> noise           ---------->        x_0    \ L1 loss
    #                                                             - x_gt   /
    
    # subtract noise from noisy x_t, and get the denoised image x_0. noise is expected to have standard std.
    # sqrt_recipm1_alphas_cumprod_t scales the noise std to to the same std used to generate the noise.
    # Eq.4: \sqrt{\bar{\alpha}_t)} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon = x_t =>
    #        x_0 = sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)   * x_t -
            extract_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
             extract_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        # When t is close to self.num_timesteps, x_start's coefficients are close to 0 (0.01~0.02), and 
        # x_t's coefficients are close to 1 (0.98~0.99).
        # This means posterior_mean of x_{t-1} is still dominated by x_t, the noisy image.
        posterior_mean = (
            extract_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # When t > 900, posterior_variance \in [0.01, 0.03].
        posterior_variance = extract_tensor(self.posterior_variance, t, x_t.shape)
        # When t > 900, posterior_log_variance_clipped \in [-5, -3].
        posterior_log_variance_clipped = extract_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes_or_embed, x_self_cond = None):
        model_output_dict = self.denoise_fn(x, t, classes_or_embed, x_self_cond)
        model_output = model_output_dict['preds_stu'][-1]

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    # p_mean_variance() returns the slightly denoised (from t to t-1) image mean and noise variance.
    def p_mean_variance(self, x, t, classes_or_embed, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, classes_or_embed, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            #x_start.clamp_(-1., 1.)
            x_start = dclamp(x_start, -1., 1.)

        # x_start is the denoised & clamped image at t=0. x is the noisy image at t.
        # When t > 900, posterior_log_variance \in [-5, -3].
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # p_sample(), p_sample_loop(), sample() are used to do inference.
    # p_sample() adds noise to the posterior image mean, according to noise variance.
    # As p_sample() is called in every iteration in p_sample_loop(), 
    # it results in a stochastic denoising trajectory,
    # i.e., may generate different images given the same input noise.
    def p_sample(self, x, t: int, classes_or_embed=None, x_self_cond = None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, classes_or_embed, x_self_cond, clip_denoised=clip_denoised)
        # no noise when t == 0
        noise = noise_like(x.shape, device, repeat_noise) if t > 0 else 0.
        # When t > 900, model_log_variance \in [-5, -3]. 
        # (0.5 * model_log_variance).exp() \in [0.10, 0.25]
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start     

    # Sampled image pixels are between [-1, 1]. Need to unnormalize_to_zero_to_one() before output.
    def p_sample_loop(self, shape, noise=None, classes_or_embed=None, 
                      clip_denoised=True, generator=None):
        batch, device = shape[0], self.betas.device

        if noise is None:
            img = fast_randn(shape, device=device, generator=generator)
        else:
            img = noise

        x_start = None

        if self.cls_embed_type == 'none':
            classes_or_embed = None
        elif classes_or_embed is None:
            # classes_or_embed is initialized as random classes.
            classes_or_embed = torch.randint(0, self.num_classes, (batch,), device=device, generator=generator)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', disable=not self.is_master):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes_or_embed, self_cond,
                                         clip_denoised=clip_denoised)

        img = unnormalize_to_zero_to_one(img)
        return img, classes_or_embed

    @torch.no_grad()
    def ddim_sample(self, shape, noise=None, classes_or_embed=None, clip_denoised=True, generator=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if noise is None:
            img = fast_randn(shape, device=device, generator=generator)
        else:
            img = noise

        x_start = None

        if self.cls_embed_type == 'none':
            classes_or_embed = None
        elif classes_or_embed is None:
            # classes_or_embed is initialized as random classes.
            classes_or_embed = torch.randint(0, self.num_classes, (batch,), device=device, generator=generator)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=not self.is_master):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes_or_embed, self_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img, classes_or_embed

    @torch.no_grad()
    def sample(self, batch_size = 16, dataset=None, generator=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        img, classes = sample_fn((batch_size, channels, image_size, image_size), generator=generator)
        # Find nearest neighbors in dataset.
        if exists(dataset):
            if exists(classes):
                # Take the first image as the representative image of each class in classes.
                nn_img_indices  = [ dataset.cls2indices[cls][0] for cls in classes.tolist() ]
                # training = False: Disable augmentation of fetched images.
                nn_img_list     = [ dataset[idx]['img_orig'] for idx in nn_img_indices ]
                nn_img = torch.stack(nn_img_list, dim=0).to(img.device)
            else:
                # Stub. Write VGG-based nn search code later.
                return None
        else:
            return None
        return img, nn_img

    def translate(self, img_orig, new_class, t_frac=0.8, generator=None):
        batch_size = img_orig.shape[0]
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        t = int(self.num_timesteps * t_frac)
        img_new, classes = sample_fn((batch_size, channels, image_size, image_size), generator=generator)
        # Find nearest neighbors in dataset.

        return img_new

    def noisy_interpolate(self, x1, x2, t_batch, w = 0.5):
        assert x1.shape == x2.shape
        # Apply the same t_batch to x1 and x2, respectively. 
        # Otherwise it's difficult to deal with different time embeddings of xt1 and xt2.
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batch), (x1, x2))

        img_interp = w * x1 + (1 - w) * x2 
        return img_interp

    @torch.no_grad()
    def image_interpolate(self, x1, x2, t = None, w = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        t_batch = torch.stack([torch.tensor(t, device=device)] * b)
        img = self.noisy_interpolate(x1, x2, t_batch, w)
        for i in reversed(range(0, t)):
            img = self.p_sample(img, i, classes=None)

        return img

    def calc_denoise1_cls_interp_loss(self, img_gt, classes, min_interp_w = 0., min_before_weight=True,
                             noise_scheme='larger_t', min_t_percentile=0.8):
        assert self.cls_embed_type != 'none' and exists(classes)

        b, device = img_gt.shape[0], img_gt.device
        assert b % 2 == 0
        b2 = b // 2
        classes1 = classes[:b2]

        # In SingletonDataset, each image is a class. So we don't have 
        # different same-class images to do interpolation.
        within_same_class = not isinstance(self.dataset, SingletonDataset)

        if within_same_class:
            img_gt1         = img_gt[:b2]
            img_gt2_dict    = self.dataset.sample_by_labels(classes1.tolist())
            img_gt2         = img_gt2_dict['img'].cuda()
            img_gt2         = normalize_to_neg_one_to_one(img_gt2)
            # Replace the second half of img_gt with randomly sampled images 
            # that are of the same classes as img_gt1.
            img_gt          = torch.cat([img_gt1, img_gt2], dim=0)
            classes         = classes1.repeat(2)

        feat_gt = extract_pre_feat(self.featnet_type, self.denoise_fn.cls_sem_feat_ext, img_gt, ref_shape=None, 
                                   has_grad=False, use_head_feat=self.denoise1_cls_sem_loss_use_head_feat)        
        feat_gt1, feat_gt2  = feat_gt[:b2], feat_gt[b2:]
        w = torch.rand((b2, ), device=img_gt.device)
        # Normalize w into [min_interp_w, 1-min_interp_w], i.e., [0.2, 0.8].
        w = (1 - 2 * min_interp_w) * w + min_interp_w
        # w_4d.shape: (b2, 1, 1, 1)
        w_4d = w.view(b2, *((1,) * (len(img_gt.shape) - 1)))
        # w_2d.shape: (b2, 1)
        w_2d = w.view(b2, 1)

        noise = fast_randn_like(img_gt)
        if noise_scheme == 'pure_noise':
            t2 = torch.full((b2, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            # as we expect the input to be pure noise, no need to interpolate two sub-batches of noisy images
            img_noisy_interp = noise[:b2]

        elif noise_scheme == 'larger_t':
            # Only use the largest 1/5 of possible t values to inject noises.
            t2 = torch.randint(int(self.num_timesteps * min_t_percentile), self.num_timesteps, (b2, ), device=device).long()
            t  = t2.repeat(2)
            img_noisy = self.q_sample(x_start=img_gt, t=t, noise=noise, distill_t_frac=-1)
            img_noisy1, img_noisy2 = img_noisy[:b2], img_noisy[b2:]
            # interpolate two sub-batches of noisy images
            img_noisy_interp = w_4d * img_noisy1 + (1 - w_4d) * img_noisy2

        elif noise_scheme == 'almost_pure_noise':
            t2 = torch.full((b2, ), self.num_timesteps - 2, device=device, dtype=torch.long)
            # self.alphas_cumprod[-1] = 0. So take -2 as the minimal alpha_cumprod, and scale it by 0.1.
            alpha_cumprod   = self.alphas_cumprod[-2] * 0.1
            alphas_cumprod  = torch.full((b, ), alpha_cumprod, device=device, dtype=img_gt.dtype)
            alphas_cumprod  = alphas_cumprod.view(b, *((1,) * (len(img_gt.shape) - 1)))
            x_start_weight  = torch.sqrt(alphas_cumprod)
            noise_weight    = torch.sqrt(1 - alphas_cumprod)
            img_noisy       = x_start_weight * img_gt + noise_weight * noise
            img_noisy1, img_noisy2 = img_noisy[:b2], img_noisy[b2:]
            # interpolate two sub-batches of noisy images
            img_noisy_interp = w_4d * img_noisy1 + (1 - w_4d) * img_noisy2

        # Embeddings of the first and second halves are the same. 
        # No need to do interpolation on class embedding.
        if within_same_class:
            cls_embed_interp = self.denoise_fn.cls_embedding(classes1)
        else:
            # Interpolate class embeddings of the two sub-batches of class embeddings.
            cls_embed = self.denoise_fn.cls_embedding(classes)
            cls_embed1, cls_embed2 = cls_embed[:b2], cls_embed[b2:]
            cls_embed_interp = w_2d * cls_embed1 + (1 - w_2d) * cls_embed2

        # Setting the last param (img_tea) to None, so that teacher module won't be executed, 
        # to reduce unnecessary compute.
        model_output_dict = self.denoise_fn(img_noisy_interp, t2, classes_or_embed=cls_embed_interp)
        img_stu_pred = model_output_dict['preds_stu'][-1]

        if self.objective == 'pred_noise':
            # img_stu_pred is the predicted noises. Subtract it from img_interp to get the predicted image.
            img_stu_pred = self.predict_start_from_noise(img_noisy_interp, t2, img_stu_pred)
        # otherwise, objective is 'pred_x0', and pred_interp is already the predicted image.
        
        if self.iter_count % 1000 == 0:
            cycle_idx = self.iter_count // 1000
            if self.local_rank <= 0:
                os.makedirs(self.sample_dir, exist_ok=True)
                # img_gt contains the two sub-batches. As nrow is half of the batch size,
                # img_gt is split into two rows, which is desired.
                img_denoise1 = torch.cat([img_gt, img_noisy_interp, img_stu_pred], dim=0)
                img_denoise1_save_path  = f'{self.sample_dir}/{cycle_idx:03}-denoise1.png'
                unnorm_save_image(img_denoise1,   img_denoise1_save_path,  nrow = b2)

        # feat_interp is the intermediate features of the denoised images. 
        # So it has to stay in the computation graph, and has_grad=True.
        feat_interp = extract_pre_feat(self.featnet_type, self.denoise_fn.cls_sem_feat_ext, img_stu_pred, ref_shape=None, 
                                       has_grad=True, use_head_feat=self.denoise1_cls_sem_loss_use_head_feat)

        loss_interp1 = self.consist_loss_fn(feat_interp, feat_gt1, reduction='none')
        loss_interp2 = self.consist_loss_fn(feat_interp, feat_gt2, reduction='none')

        if min_before_weight:
            # if neighbor_mask[i, pos] = 1, i.e., this pixel's feature is more similar to sub-batch1, 
            # then use loss weight w. Otherwise, it's more similar to sub-batch2, use loss weight 1-w.
            neighbor_mask = (loss_interp1 < loss_interp2).float()
            loss_weight = neighbor_mask * w_4d + (1 - neighbor_mask) * (1 - w_4d)
            # The more similar features from tea_feat of either sub-batch1 or sub-batch2 are selected 
            # to compute the loss with feat_interp at each pixel.
            loss_interp = torch.minimum(loss_interp1, loss_interp2) * loss_weight
            loss_interp = loss_interp.mean()
        else:
            loss_interp1_weighted = loss_interp1 * w_4d
            loss_interp2_weighted = loss_interp2 * (1 - w_4d)
            neighbor_mask = (loss_interp1_weighted < loss_interp2_weighted).float()
            sel_weight = neighbor_mask * w_4d + (1 - neighbor_mask) * (1 - w_4d)
            total_weight = sel_weight.sum() + 1e-6
            loss_interp = torch.minimum(loss_interp1_weighted, loss_interp2_weighted)
            loss_interp = loss_interp.sum() / total_weight

        return loss_interp

    # cls_single_loss only considers the semantic consistency of the denoised images 
    # with the ground truth images, taking the class embeddings to guide the semantics of generated images.
    def calc_denoise1_cls_single_loss(self, img_gt, classes, noise_scheme='larger_t', min_t_percentile=0.8):
        assert self.cls_embed_type != 'none' and exists(classes)

        b, device = img_gt.shape[0], img_gt.device
        # assert b % 2 == 0
        b2 = b # // 2
        img_gt2 = img_gt[:b2]
        classes2 = classes[:b2]

        feat_gt = extract_pre_feat(self.featnet_type, self.denoise_fn.cls_sem_feat_ext, img_gt2, ref_shape=None, 
                                   has_grad=False, use_head_feat=self.denoise1_cls_sem_loss_use_head_feat)        
        noise   = fast_randn_like(img_gt2)

        if noise_scheme == 'pure_noise':
            t = torch.full((b2, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            img_noisy = noise

        elif noise_scheme == 'larger_t':
            # Only use the largest 1/5 of possible t values to inject noises.
            t = torch.randint(int(self.num_timesteps * min_t_percentile), self.num_timesteps, (b2, ), device=device).long()
            img_noisy = self.q_sample(x_start=img_gt2, t=t, noise=noise, distill_t_frac=-1)

        elif noise_scheme == 'almost_pure_noise':
            t = torch.full((b2, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            # self.alphas_cumprod[-1] = 0. So take -2 as the minimal alpha_cumprod, and scale it by 0.1.
            alpha_cumprod   = self.alphas_cumprod[-2] * 0.1
            alphas_cumprod  = torch.full((b2, ), alpha_cumprod, device=device, dtype=img_gt.dtype)
            alphas_cumprod  = alphas_cumprod.view(b2, *((1,) * (len(img_gt.shape) - 1)))
            x_start_weight  = torch.sqrt(alphas_cumprod)
            noise_weight    = torch.sqrt(1 - alphas_cumprod)
            img_noisy       = x_start_weight * img_gt2 + noise_weight * noise

        cls_embed = self.denoise_fn.cls_embedding(classes2)

        # Set img_tea to None, so that teacher module won't be executed and trained, 
        # to reduce unnecessary compute.
        # Shouldn't train the teacher using class semantics loss, as the teacher is specialized 
        # to handle easier (less noisy) images.
        model_output_dict = self.denoise_fn(img_noisy, t, classes_or_embed=cls_embed, img_tea=None)
        img_stu_pred = model_output_dict['preds_stu'][-1]

        if self.objective == 'pred_noise':
            # img_stu_pred is the predicted noises. Subtract it from img_noisy to get the predicted image.
            img_stu_pred = self.predict_start_from_noise(img_noisy, t, img_stu_pred)
        # otherwise, objective is 'pred_x0', and img_stu_pred is already the predicted image.

        if self.iter_count % 1000 == 0:
            cycle_idx = self.iter_count // 1000
            if self.local_rank <= 0:
                os.makedirs(self.sample_dir, exist_ok=True)
                img_denoise1 = torch.cat([img_gt2, img_noisy, img_stu_pred], dim=0)
                img_denoise1_save_path  = f'{self.sample_dir}/{cycle_idx:03}-denoise1.png'
                unnorm_save_image(img_denoise1,   img_denoise1_save_path,  nrow = b2)

        # feat_stu is the intermediate features of the denoised images. 
        # So it has to stay in the computation graph, and has_grad=True.
        feat_stu = extract_pre_feat(self.featnet_type, self.denoise_fn.cls_sem_feat_ext, img_stu_pred, ref_shape=None, 
                                    has_grad=True, use_head_feat=self.denoise1_cls_sem_loss_use_head_feat)

        loss_cls_sem = self.consist_loss_fn(feat_stu, feat_gt)
        return loss_cls_sem

    # inject random noise into x_start. sqrt_one_minus_alphas_cumprod_t is the std of the noise.
    def q_sample(self, x_start, t, noise=None, distill_t_frac=-1):
        assert distill_t_frac <= 1

        noise = default(noise, lambda: fast_randn_like(x_start))
        #x_start_weight = extract_tensor(self.sqrt_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)
        #noise_weight   = extract_tensor(self.sqrt_one_minus_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)
        # t serves as a tensor of indices, to extract elements from alphas_cumprod.
        alphas_cumprod  = extract_tensor(self.alphas_cumprod, t, x_start.shape)
        x_start_weight  = torch.sqrt(alphas_cumprod)
        noise_weight    = torch.sqrt(1 - alphas_cumprod)
        x_noisy1 = x_start_weight * x_start + noise_weight * noise

        if self.debug and self.iter_count >=0 and self.iter_count < 10:
            print0(f'{self.iter_count} x_start_weight\n{x_start_weight.flatten()}')
            print0(f'{self.iter_count} noise_weight\n{noise_weight.flatten()}')

        # Conventional sampling. Do not sample for an easier t.
        if distill_t_frac == -1:
            return x_noisy1
            
        else:
            # No noise is added to x_noisy2.
            if distill_t_frac == 0:
                x_noisy2 = x_start
            else:
                # Do conventional sampling, as well as an easier x2 according to a t2 < t.
                t2 = (t * distill_t_frac).long()
                alphas_cumprod2 = extract_tensor(self.alphas_cumprod, t2, x_start.shape)
                x_start_weight2 = torch.sqrt(alphas_cumprod2)
                noise_weight2   = torch.sqrt(1 - alphas_cumprod2)
                x_noisy2 = x_start_weight2 * x_start + noise_weight2 * noise

                if self.debug and self.iter_count >=0 and self.iter_count < 10:
                    print0(f'{self.iter_count} x_start_weight2\n{x_start_weight2.flatten()}')
                    print0(f'{self.iter_count} noise_weight2\n{noise_weight2.flatten()}')

            return x_noisy1, x_noisy2

    # LapLoss doesn't work.
    def laploss(self, x_pred, x):
        lap_loss = self.laploss_fun(x_pred, x)
        l1_loss  = F.l1_loss(x_pred, x)
        # print(f'lap_loss: {lap_loss.item()}, l1_loss: {l1_loss.item()}')
        if torch.isnan(lap_loss):
            lap_loss = 0
        else:
            lap_loss = lap_loss / lap_loss.item()
        if l1_loss > 1:
            l1_loss = l1_loss / l1_loss.item()
        return 0.2 * lap_loss + 0.8 * l1_loss

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'lap':
            return self.laploss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    # For multi-scale training. The pred images from a middle block is smaller 
    # than the ground truth images. Therefore manually upsample the pred images.
    def loss_fn_alignsize(self, pred, target):
        if target.shape != pred.shape:
            target = F.interpolate(target, pred.shape[2:], mode='bilinear', align_corners=False)
        return self.loss_fn(pred, target)

    # compute the consistency loss. consist_loss_fn is a function, instead of property as loss_fn.
    def consist_loss_fn(self, feat_pred, feat_gt, reduction='mean'):
        if self.consist_loss_type == 'l1':
            return F.l1_loss(feat_pred, feat_gt, reduction=reduction)
        elif self.consist_loss_type == 'cosine':
            # Assume feat_pred and feat_gt are of [B, C, 1, 1].
            feat_pred   = feat_pred.squeeze(3).squeeze(2)
            feat_gt     = feat_gt.squeeze(3).squeeze(2)
            target      = torch.ones(feat_gt.shape[0], device=feat_gt.device)
            return F.cosine_embedding_loss(feat_pred, feat_gt, target, reduction=reduction)
        else:
            raise ValueError(f'invalid consistency loss type {self.consist_loss_type}')

    # x_start: initial image. x_orig: the same images without augmentation.
    def p_losses(self, x_start, x_orig, t, classes, noise = None):
        b, c, h, w = x_start.shape
        # noise: a Gaussian noise for each pixel.
        noise = default(noise, lambda: fast_randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, 
                                distill_t_frac=self.distill_t_frac)
        # Sample an easier x2 according to a smaller t.
        if self.distill_t_frac == -1:
            x       = x_noisy
            x_tea   = None
        else:
            x, x_tea = x_noisy

        if self.debug and self.iter_count < 10:
            unnorm_save_image(x, f'{self.sample_dir}/{self.iter_count}-stu.png')
            if exists(x_tea):
                unnorm_save_image(x_tea, f'{self.sample_dir}/{self.iter_count}-tea.png')

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # https://github.com/lucidrains/denoising-diffusion-pytorch/commit/689593a5792512f81c159f7f441afcd2fcc26492
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, classes_or_embed=classes).pred_x_start
                x_self_cond.detach_()

        model_output_dict    = self.denoise_fn(x, t, classes_or_embed=classes, 
                                               x_self_cond=x_self_cond, img_tea=x_tea)
        preds_stu, preds_tea = model_output_dict['preds_stu'],  model_output_dict['preds_tea']
        noise_feat, tea_feat = model_output_dict['noise_feat'], model_output_dict['tea_feat']

        loss_stu, loss_tea, loss_align_tea_stu = torch.zeros(3, device=x_start.device)

        # Compute the loss for each scale of the output image.
        for i, pred_stu in enumerate(preds_stu):
            if self.objective == 'pred_noise':
                # Laplacian loss doesn't help. Instead, it makes convergence very slow. So it's removed.
                target = noise
            elif self.objective == 'pred_x0':
                target = x_start
            else:
                raise ValueError(f'unknown objective {self.objective}')

            loss_stu = loss_stu + self.loss_fn_alignsize(pred_stu, target)
            if self.distillation_type != 'none' and exists(preds_tea):
                pred_tea    = preds_tea[i]
                loss_tea    = loss_tea + self.loss_fn_alignsize(pred_tea, target)
                loss_align_tea_stu = loss_align_tea_stu + F.l1_loss(noise_feat, tea_feat.detach())

        loss_stu, loss_tea, loss_align_tea_stu = loss_stu / len(preds_stu), \
                                                 loss_tea / len(preds_stu), \
                                                 loss_align_tea_stu / len(preds_stu)

        if self.denoise1_cls_sem_loss_type != 'none':
            if self.denoise1_cls_sem_loss_type == 'single':
                loss_cls_sem = self.calc_denoise1_cls_single_loss(x_start, classes)
            elif self.denoise1_cls_sem_loss_type == 'interp':
                loss_cls_sem = self.calc_denoise1_cls_interp_loss(x_start, classes)
        else:
            loss_cls_sem = torch.zeros_like(loss_stu)

        loss = loss_stu + loss_tea + \
                self.align_tea_stu_feat_weight * loss_align_tea_stu + \
                self.denoise1_cls_sem_loss_weight * loss_cls_sem

        # Capping loss at 1 might helps early iterations when losses are unstable (occasionally very large).
        if loss > 1:
            loss = loss / loss.item()
        return { 'loss': loss, 'loss_stu': loss_stu, 'loss_tea': loss_tea, 'loss_cls_sem': loss_cls_sem }

    def forward(self, img, img_orig, classes, iter_count, *args, **kwargs):
        self.iter_count = iter_count
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        if not (h == img_size and w == img_size):
            print0(f'height and width of image must be {img_size}')
            breakpoint()
        
        # t: random numbers of steps between 0 and num_timesteps - 1 (num_timesteps default is 1000)
        # (b,): different random steps for different images in a batch.
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        img_orig = normalize_to_neg_one_to_one(img_orig)
        return self.p_losses(img, img_orig, t, classes, *args, **kwargs)

    def sample_t(self, iter_count, max_iter, b, device, min_temp=0, max_temp=1.5):
        temperature = min_temp + (max_temp - min_temp) * (iter_count / max_iter)
        weight = torch.arange(1, b+1, device=device).float() ** temperature

        if temperature == 0:
            return torch.randint(0, self.num_timesteps, (b, ), device=device).long()
        else:
            return torch.multinomial(weight, (b, ), replacement=True).long()
