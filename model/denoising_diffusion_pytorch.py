from doctest import NORMALIZE_WHITESPACE
import math
import torch
from torch import nn, einsum
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import utils
from functools import partial

import timm
from einops import rearrange
from .laplacian import LapLoss
from .utils import timm_extract_features, print0, exists, exists_add, repeat_interleave, \
                   default, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, unnorm_save_image, \
                   SimpleDataset
import os

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

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules. Kernel size is fixed as 3.
# No downsampling is done in Block, i.e., it outputs a tensor of the same (H,W) as the input.
class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size, padding = (kernel_size-1)//2 ),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

# Different ResnetBlock's have different mlp (time embedding module).
# No downsampling is done in ResnetBlock.
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size=3, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),      # Sigmoid Linear Unit, aka swish. https://pytorch.org/docs/stable/_images/SiLU.png
            nn.Linear(time_emb_dim, dim_out),
            nn.LayerNorm(dim_out)       # Newly added.
        ) if exists(time_emb_dim) else None

        # block1 and block2 are ended with a group norm and a SiLU activation.
        self.block1 = Block(dim,     dim_out, kernel_size = kernel_size, groups = groups)
        self.block2 = Block(dim_out, dim_out, kernel_size = kernel_size, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # time embedding is incorporated between block1 and block2.
    def forward(self, x, time_emb = None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb).permute(0, 3, 1, 2)
            if time_emb.shape[2] > 1 and time_emb.shape[3] > 1:
                time_emb = repeat_interleave(repeat_interleave(time_emb, h.shape[2] // time_emb.shape[2], dim=2), 
                                             h.shape[3] // time_emb.shape[3], dim=3)

            h = h + time_emb

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
            self.memory = nn.Parameter(torch.randn(1, dim, 1, memory_size))

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
    def __init__(self, dim, heads = 4, dim_head = 32, memory_size=0):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.memory_size = memory_size
        if self.memory_size > 0:
            # Persistent memory that serves as a codebook used for image generation.
            self.memory = nn.Parameter(torch.randn(1, dim, 1, memory_size))

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
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
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
        with_time_emb = True,
        resnet_block_groups = 8,
        memory_size = 1024,
        num_classes = -1,
        learned_variance = False,
        featnet_type = 'none',
        finetune_tea_feat_ext = False,
        distillation_type = 'none',
        cls_embed_type = 'none',
    ):
        super().__init__()

        # number of input channels
        self.channels = channels

        # dim = 64, init_dim -> 42.
        init_dim = default(init_dim, dim // 3 * 2)
        # image size is the same after init_conv, as default stride=1.
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        # init_conv + num_resolutions (4) layers.
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 4 pairs of layers in the encoder/decoder.
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        # Fixed sinosudal embedding, transformed by two linear layers.
        if with_time_emb:
            time_dim = dim * 4              # 256
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
                nn.LayerNorm(time_dim),        # Newly added
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.num_classes    = num_classes
        # It's possible that self.num_classes < 0, i.e., the number of classes is not provided.
        # In this case, we set cls_embed_type to 'none'.
        if self.num_classes <= 0:
            self.cls_embed_type = 'none'
        else:
            self.cls_embed_type = cls_embed_type

        if self.cls_embed_type != 'none':
            self.cls_embedding = nn.Embedding(self.num_classes, time_dim)
            self.cls_embed_ln  = nn.LayerNorm(time_dim)
            print0("cls_embedding: ", list(self.cls_embedding.weight.shape))
        else:
            self.cls_embedding = None
            self.cls_embed_ln = None
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        # ups_tea: teacher decoder
        self.ups_tea = nn.ModuleList([])


        # LinearAttention layers are used in encoder/decoder.
        # Using vanilla attention in encoder/decoder takes 13x RAM, and is 3x slower.

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                # A block_klass is a two-layer conv with residual connection. 
                # block_klass doesn't downsample features.
                block_klass(dim_in,  dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                # att(norm(x)) + x.
                # Seems adding memory to encoder hurts performance: image features poluted by memory?
                Residual(PreNorm(dim_out, LinearAttention(dim_out, memory_size=0))),
                # downsampling is done with a 4x4 kernel, stride-2 conv.
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        # Seems the memory in mid_attn doesn't make much difference.
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim, memory_size=0)))
        # Seems setting kernel size to 1 leads to slightly worse images?
        self.mid_block2 = block_klass(mid_dim, mid_dim, kernel_size=1, time_emb_dim = time_dim)

        self.featnet_type      = featnet_type
        self.finetune_tea_feat_ext  = finetune_tea_feat_ext
        self.distillation_type = distillation_type

        if self.featnet_type != 'none' and self.featnet_type != 'mini':
            # Tried 'efficientnet_b0', but it doesn't perform well.
            # 'resnet34', 'resnet18', 'repvgg_b0'
            self.featnet_type   = self.featnet_type 
            self.featnet_dim    = timm_model2dim[self.featnet_type]
            self.consistency_feat_ext    = timm.create_model(self.featnet_type, pretrained=True)

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

        # distillation_type: 'none' or 'tfrac'. 
        if self.distillation_type == 'none':
            extra_up_dim = 0
        # tfrac: teacher sees less-noisy images (either sees a miniature or through a feature network)
        else:
            if self.featnet_type == 'mini':
                extra_up_dim = 3
            else:
                # number of output features from the image feature extractor.
                extra_up_dim = self.featnet_dim
        
        extra_up_dims = [ extra_up_dim ] + [0] * (num_resolutions - 1)
                        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2 + extra_up_dims[ind], dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, memory_size=memory_size))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Sharing ups seems to perform worse.
        self.tea_share_ups = False
        if self.tea_share_ups:
            self.ups_tea = self.ups
        else:
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (num_resolutions - 1)
                # miniature image / image features as teacher's priviliged information.

                self.ups_tea.append(nn.ModuleList([
                    block_klass(dim_out * 2 + extra_up_dims[ind], dim_in, time_emb_dim = time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in, memory_size=memory_size))),
                    Upsample(dim_in) if not is_last else nn.Identity()
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, self.out_dim, 1)
        )

    # extract_pre_feat(): extract features using a pretrained model.
    # use_head_feat: use the features that feat_extractor uses to do the final classification. 
    # It's not relevant to class embeddings used in as part of Unet features.
    def extract_pre_feat(self, feat_extractor, img, ref_shape, has_grad=True, use_head_feat=False):
        if self.featnet_type == 'none':
            return None

        if self.featnet_type == 'mini':
            # A miniature image as teacher's priviliged information. 
            # 128x128 images will be resized to 16x16 below.
            distill_feat = img
        else:
            if has_grad:
                distill_feat = timm_extract_features(feat_extractor, img, use_head_feat)
            else:
                # Wrap the feature extraction with no_grad() to save RAM.
                with torch.no_grad():
                    distill_feat = timm_extract_features(feat_extractor, img, use_head_feat)                

        if exists(ref_shape):
            # For 128x128 images, vit features are 4x4. Resize to 16x16.
            distill_feat = F.interpolate(distill_feat, size=ref_shape, mode='bilinear', align_corners=False)
        # Otherwise, do not resize distill_feat.
        return distill_feat

    def forward(self, x, time, classes=None, cls_embed=None, img_tea=None):
        init_noise = x
        x = self.init_conv(x)
        # t: time embedding.
        if exists(self.time_mlp):
            t = self.time_mlp(time.flatten())
            # t: [batch, 1, 1, time_dim=256].
            t = t.view(time.shape[0], *((1,) * (len(x.shape) - 2)), -1)
        else:
            t = None

        if self.cls_embed_type != 'none':
            # If classes is provided, cls_embed should be None.
            # If cls_embed is provided, classes should be None (but we didn't check it yet).
            if exists(classes):
                assert cls_embed is None
                cls_embed = self.cls_embedding(classes)
            
            cls_embed = self.cls_embed_ln(cls_embed)
            # cls_embed: [batch, 1, 1, time_dim=256].
            cls_embed = cls_embed.view(cls_embed.shape[0], *((1,) * (len(x.shape) - 2)), -1)
            # The teacher always sees the class embedding.
            t_tea = exists_add(t, cls_embed)
            # 'tea_stu': Both the student and teacher see the class embedding.
            t_stu = t_tea if self.cls_embed_type == 'tea_stu' else t
        else:
            t_stu = t_tea = t

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
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
            noise_feat = self.extract_pre_feat(self.dist_feat_ext_stu, init_noise, 
                                               mid_feat.shape[2:], 
                                               has_grad=True)
        else:
            noise_feat = None

        for ind, (block1, block2, attn, upsample) in enumerate(self.ups):
            if ind == 0 and exists(noise_feat):
                x = torch.cat((x, h[-ind-1], noise_feat), dim=1)
            else:
                x = torch.cat((x, h[-ind-1]), dim=1)

            x = block1(x, t_stu)
            x = block2(x, t_stu)
            x = attn(x)
            # All blocks above don't change feature resolution. Only upsample explicitly here.
            # upsample() is a 4x4 conv, stride-2 transposed conv.
            x = upsample(x)

        pred_stu = self.final_conv(x)

        # img_tea is provided. Do distillation.
        if self.distillation_type == 'tfrac' and exists(img_tea):
            # finetune_tea_feat_ext controls whether to fine-tune the teacher feature extractor.
            tea_feat = self.extract_pre_feat(self.dist_feat_ext_tea, img_tea, 
                                             mid_feat.shape[2:], 
                                             has_grad=self.finetune_tea_feat_ext)
        else:
            tea_feat = None

        if self.distillation_type != 'none' and exists(tea_feat):
            x = mid_feat
            for ind, (block1, block2, attn, upsample) in enumerate(self.ups_tea):
                if ind == 0 and exists(tea_feat):
                    x = torch.cat((x, h[-ind-1], tea_feat), dim=1)
                else:
                    x = torch.cat((x, h[-ind-1]), dim=1)

                x = block1(x, t_tea)
                x = block2(x, t_tea)
                x = attn(x)
                # All blocks above don't change feature resolution. Only upsample explicitly here.
                # upsample() is a 4x4 conv, stride-2 transposed conv.
                x = upsample(x)

            pred_tea = self.final_conv(x)

        else:
            pred_tea = None

        return { 'pred_stu': pred_stu, 'pred_tea': pred_tea, 'noise_feat': noise_feat, 'tea_feat': tea_feat }

# gaussian diffusion trainer class
# suppose t is a 1-d tensor of indices.
def extract_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # out: [b, 1, 1, 1]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(num_timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_alpha_schedule(num_timesteps):
    steps = num_timesteps + 1
    alphas_cumprod = torch.linspace(1, 0, steps, dtype = torch.float64)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# Original schedule used in https://github.com/hojonathanho/diffusion
def linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    return betas

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        num_timesteps = 1000,
        alpha_beta_schedule = 'linb',
        loss_type = 'l1',
        consist_loss_type = 'cosine',
        objective = 'pred_noise',
        featnet_type = 'none',
        distillation_type = 'none',
        distill_t_frac = 0.,
        cls_embed_type = 'none',
        num_classes = -1,
        dataset = None,
        consistency_use_head_feat = True,
        cls_guide_type = 'none',
        cls_guide_loss_weight = 0.01,
        align_tea_stu_feat_weight = 0,
        sample_dir = 'samples',      
        debug = False,
        sampleseed = 5678,
    ):
        super().__init__()
        if isinstance(denoise_fn, torch.nn.DataParallel):
            denoise_fn_channels = denoise_fn.module.channels
            denoise_fn_out_dim  = denoise_fn.module.out_dim
        else:
            denoise_fn_channels = denoise_fn.channels
            denoise_fn_out_dim  = denoise_fn.out_dim

        assert not (type(self) == GaussianDiffusion and denoise_fn_channels != denoise_fn_out_dim)

        self.channels = channels
        self.image_size         = image_size
        self.denoise_fn         = denoise_fn    # Unet
        self.objective          = objective
        self.featnet_type       = featnet_type
        self.distillation_type  = distillation_type
        self.distill_t_frac     = distill_t_frac if (self.distillation_type == 'tfrac') else -1
        self.cls_embed_type     = cls_embed_type
        self.num_classes        = num_classes
        self.dataset            = dataset
        # It's possible that self.num_classes < 0, i.e., the number of classes is not provided.
        # In this case, we set cls_embed_type to 'none'.
        if self.num_classes <= 0:
            self.cls_embed_type = 'none'

        # interpolation loss reduces performance.
        self.interp_loss_weight     = 0
        # Use backbone head features for consistency losses, i.e., with the geometric dimensions collapsed.
        # such as class-guidance loss and interpolation loss (interpolation doesn't work well, though).
        self.consistency_use_head_feat  = consistency_use_head_feat
        self.cls_guide_type             = cls_guide_type
        self.cls_guide_loss_weight      = cls_guide_loss_weight
        self.align_tea_stu_feat_weight  = align_tea_stu_feat_weight
        self.sample_dir = sample_dir
        self.debug = debug
        self.num_timesteps      = num_timesteps
        self.alpha_beta_schedule = alpha_beta_schedule
        self.iter_count = 0

        # Sampling uses independent noises and random seeds from training.
        old_rng_state = torch.get_rng_state()
        torch.manual_seed(sampleseed)
        self.sample_rng_state = torch.get_rng_state()
        torch.set_rng_state(old_rng_state)

        if self.alpha_beta_schedule == 'cosb':
            print0("Use cosine_beta_schedule")
            betas = cosine_beta_schedule(self.num_timesteps)
        elif self.alpha_beta_schedule == 'lina':
            print0("Use linear_alpha_schedule")
            betas = linear_alpha_schedule(self.num_timesteps)
        elif self.alpha_beta_schedule == 'linb':
            print0("Use linear_beta_schedule")
            betas = linear_beta_schedule(self.num_timesteps)            
        else:
            breakpoint()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.loss_type = loss_type
        self.consist_loss_type = consist_loss_type
        self.laploss_fun    = LapLoss()
        
        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # subtract noise from noisy x_t, and get the denoised image. noise is expected to have standard std.
    # sqrt_recipm1_alphas_cumprod_t scales the noise std to to the same std used to generate the noise.
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)   * x_t -
            extract_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # p_mean_variance() returns the denoised/generated image mean and noise variance.
    def p_mean_variance(self, x, t, classes, clip_denoised: bool):
        model_output_dict = self.denoise_fn(x, t, classes=classes)
        model_output = model_output_dict['pred_stu']

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    # p_sample(), p_sample_loop(), sample() are used to do inference.
    # p_sample() adds noise to image mean, according to noise variance.
    # As p_sample() is called in every iteration in p_sample_loop(), 
    # it results in a stochastic denoising trajectory,
    # i.e., may generate different images given the same input noise.
    @torch.no_grad()
    def p_sample(self, x, t, classes=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, classes=classes, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        if self.cls_embed_type == 'none':
            classes = None
        else:
            classes = torch.randint(0, self.num_classes, (b,), device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), classes=classes)

        img = unnormalize_to_zero_to_one(img)
        return img, classes

    @torch.no_grad()
    def sample(self, batch_size = 16, dataset=None):
        image_size = self.image_size
        channels = self.channels
        img, classes = self.p_sample_loop((batch_size, channels, image_size, image_size))
        # Find nearest neighbors in dataset.
        if exists(dataset):
            if exists(classes):
                # Take the first image as the representative image of each class in classes.
                nn_img_indices = [ dataset.cls2indices[cls][0] for cls in classes ]
                old_training_status = dataset.training
                # training = False: Disable augmentation of fetched images.
                dataset.training = False
                nn_img_list = [ dataset[idx]['img'] for idx in nn_img_indices ]
                dataset.training = old_training_status
                nn_img = torch.stack(nn_img_list, dim=0).to(img.device)
            else:
                # Stub. Write VGG-based nn search code later.
                return None
        else:
            return None
        return img, nn_img

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
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), classes=None)

        return img

    def calc_cls_interp_loss(self, img_gt, img_orig, classes, min_interp_w = 0., min_before_weight=True,
                             noise_scheme='larger_t', min_t_percentile=0.75):
        assert self.cls_embed_type != 'none' and exists(classes)

        b, device = img_gt.shape[0], img_gt.device
        assert b % 2 == 0
        b2 = b // 2
        classes1 = classes[:b2]

        # In SimpleDataset, each image is a class. So we don't have 
        # different same-class images to do interpolation.
        within_same_class = not isinstance(self.dataset, SimpleDataset)

        if within_same_class:
            img_gt1         = img_gt[:b2]
            img_gt2_dict    = self.dataset.sample_by_labels(classes1)
            img_gt2         = img_gt2_dict['img'].cuda()
            img_gt2         = normalize_to_neg_one_to_one(img_gt2)
            # Replace the second half of img_gt with randomly sampled images 
            # that are of the same classes as img_gt1.
            img_gt          = torch.cat([img_gt1, img_gt2], dim=0)
            img_orig1       = img_orig[:b2]
            img_orig2       = img_gt2_dict['img_orig'].cuda()
            img_orig2       = normalize_to_neg_one_to_one(img_orig2)
            img_orig        = torch.cat([img_orig1, img_orig2], dim=0)
            classes         = classes1.repeat(2)

        feat_gt = self.denoise_fn.extract_pre_feat(self.denoise_fn.consistency_feat_ext, img_gt, ref_shape=None, 
                                                   has_grad=True, use_head_feat=self.consistency_use_head_feat)        
        feat_gt1, feat_gt2  = feat_gt[:b2], feat_gt[b2:]
        w = torch.rand((b2, ), device=img_gt.device)
        # Normalize w into [min_interp_w, 1-min_interp_w], i.e., [0.2, 0.8].
        w = (1 - 2 * min_interp_w) * w + min_interp_w
        # w.shape: (b2, 1, 1, 1)
        w = w.view(b2, *((1,) * (len(img_gt.shape) - 1)))

        noise = torch.randn_like(img_gt)
        if noise_scheme == 'pure_noise':
            t2 = torch.full((b2, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            img_noisy_interp = noise[:b2]

        elif noise_scheme == 'larger_t':
            # Only use the largest 1/4 of possible t values to inject noises.
            t2 = torch.randint(int(self.num_timesteps * min_t_percentile), self.num_timesteps, (b2, ), device=device).long()
            t  = t2.repeat(2)
            img_noisy = self.q_sample(x_start=img_gt, t=t, noise=noise, distill_t_frac=-1)
            img_noisy1, img_noisy2 = img_noisy[:b2], img_noisy[b2:]
            img_noisy_interp = w * img_noisy1 + (1 - w) * img_noisy2

        elif noise_scheme == 'almost_pure_noise':
            t2 = torch.full((b2, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            # self.alphas_cumprod[-1] = 0. So take -2 as the minimal alpha_cumprod, and scale it by 0.1.
            alpha_cumprod   = self.alphas_cumprod[-2] * 0.1
            alphas_cumprod  = torch.full((b, ), alpha_cumprod, device=device, dtype=img_gt.dtype)
            alphas_cumprod  = alphas_cumprod.view(b, *((1,) * (len(img_gt.shape) - 1)))
            x_start_weight  = torch.sqrt(alphas_cumprod)
            noise_weight    = torch.sqrt(1 - alphas_cumprod)
            img_noisy       = x_start_weight * img_gt + noise_weight * noise
            img_noisy1, img_noisy2 = img_noisy[:b2], img_noisy[b2:]
            img_noisy_interp = w * img_noisy1 + (1 - w) * img_noisy2

        # Embeddings of the first and second halves are the same. 
        # No need to do interpolation on class embedding.
        if within_same_class:
            cls_embed_interp = self.denoise_fn.cls_embedding(classes1)
        else:
            cls_embed = self.denoise_fn.cls_embedding(classes)
            cls_embed = cls_embed.view(b, *((1,) * (len(img_gt.shape) - 2)), -1)
            cls_embed1, cls_embed2 = cls_embed[:b2], cls_embed[b2:]
            cls_embed_interp = w * cls_embed1 + (1 - w) * cls_embed2

        # Setting the last param (img_tea) to None, so that teacher module won't be executed, 
        # to reduce unnecessary compute.
        model_output_dict = self.denoise_fn(img_noisy_interp, t2, cls_embed=cls_embed_interp)
        img_stu_pred = model_output_dict['pred_stu']

        if self.objective == 'pred_noise':
            # img_stu_pred is the predicted noises. Subtract it from img_interp to get the predicted image.
            img_stu_pred = self.predict_start_from_noise(img_noisy_interp, t2, img_stu_pred)
        # otherwise, objective is 'pred_x0', and pred_interp is already the predicted image.

        if self.iter_count % 1000 == 0:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            sample_dir = f'{self.sample_dir}/interp'
            os.makedirs(sample_dir, exist_ok=True)
            img_gtaug_save_path  = f'{sample_dir}/{self.iter_count}-{local_rank}-aug.png'
            unnorm_save_image(img_gt,   img_gtaug_save_path,  nrow = 8)
            #img_gtorig_save_path = f'{sample_dir}/{self.iter_count}-{local_rank}-orig.png'
            #unnorm_save_image(img_orig, img_gtorig_save_path, nrow = 8)

            #print("GT images for interpolation are saved to", img_gt_save_path)
            img_noisy_save_path = f'{sample_dir}/{self.iter_count}-{local_rank}-noisy.png'
            unnorm_save_image(img_noisy_interp, img_noisy_save_path, nrow = 8)
            #print("Noisy images for interpolation are saved to", img_noisy_save_path)
            img_pred_save_path  = f'{sample_dir}/{self.iter_count}-{local_rank}-pred.png'
            unnorm_save_image(img_stu_pred, img_pred_save_path, nrow = 8)
            #print("Predicted images are saved to", img_pred_save_path)

        feat_interp = self.denoise_fn.extract_pre_feat(self.denoise_fn.consistency_feat_ext, img_stu_pred, ref_shape=None, 
                                                       has_grad=True, use_head_feat=self.consistency_use_head_feat)

        loss_interp1 = self.consist_loss_fn(feat_interp, feat_gt1, reduction='none')
        loss_interp2 = self.consist_loss_fn(feat_interp, feat_gt2, reduction='none')

        if min_before_weight:
            # if neighbor_mask[i, pos] = 1, i.e., this pixel's feature is more similar to sub-batch1, 
            # then use loss weight w. Otherwise, it's more similar to sub-batch2, use loss weight 1-w.
            neighbor_mask = (loss_interp1 < loss_interp2).float()
            loss_weight = neighbor_mask * w + (1 - neighbor_mask) * (1 - w)
            # The more similar features from tea_feat of either sub-batch1 or sub-batch2 are selected 
            # to compute the loss with feat_interp at each pixel.
            loss_interp = torch.minimum(loss_interp1, loss_interp2) * loss_weight
            loss_interp = loss_interp.mean()
        else:
            loss_interp1_weighted = loss_interp1 * w
            loss_interp2_weighted = loss_interp2 * (1 - w)
            neighbor_mask = (loss_interp1_weighted < loss_interp2_weighted).float()
            sel_weight = neighbor_mask * w + (1 - neighbor_mask) * (1 - w)
            total_weight = sel_weight.sum() + 1e-6
            loss_interp = torch.minimum(loss_interp1_weighted, loss_interp2_weighted)
            loss_interp = loss_interp.sum() / total_weight

        return loss_interp

    def calc_cls_single_loss(self, img_gt, img_orig, classes, noise_scheme='larger_t', min_t_percentile=0.9):
        assert self.cls_embed_type != 'none' and exists(classes)

        b, device = img_gt.shape[0], img_gt.device
        feat_gt = self.denoise_fn.extract_pre_feat(self.denoise_fn.consistency_feat_ext, img_gt, ref_shape=None, 
                                                   has_grad=True, use_head_feat=self.consistency_use_head_feat)        
        noise = torch.randn_like(img_gt)

        if noise_scheme == 'pure_noise':
            t = torch.full((b, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            img_noisy = noise

        elif noise_scheme == 'larger_t':
            # Only use the largest 1/10 of possible t values to inject noises.
            t = torch.randint(int(self.num_timesteps * min_t_percentile), self.num_timesteps, (b, ), device=device).long()
            img_noisy = self.q_sample(x_start=img_gt, t=t, noise=noise, distill_t_frac=-1)

        elif noise_scheme == 'almost_pure_noise':
            t = torch.full((b, ), self.num_timesteps - 1, device=device, dtype=torch.long)
            # self.alphas_cumprod[-1] = 0. So take -2 as the minimal alpha_cumprod, and scale it by 0.1.
            alpha_cumprod   = self.alphas_cumprod[-2] * 0.1
            alphas_cumprod  = torch.full((b, ), alpha_cumprod, device=device, dtype=img_gt.dtype)
            alphas_cumprod  = alphas_cumprod.view(b, *((1,) * (len(img_gt.shape) - 1)))
            x_start_weight  = torch.sqrt(alphas_cumprod)
            noise_weight    = torch.sqrt(1 - alphas_cumprod)
            img_noisy       = x_start_weight * img_gt + noise_weight * noise

        cls_embed = self.denoise_fn.cls_embedding(classes)
        cls_embed = cls_embed.view(b, *((1,) * (len(img_noisy.shape) - 2)), -1)

        # Set img_tea to None, so that teacher module won't be executed and trained, 
        # to reduce unnecessary compute.
        # Shouldn't train the teacher using class guidance, as the teacher is specialized 
        # to handle easier (less noisy) images.
        model_output_dict = self.denoise_fn(img_noisy, t, cls_embed=cls_embed, img_tea=None)
        img_stu_pred = model_output_dict['pred_stu']

        if self.objective == 'pred_noise':
            # img_stu_pred is the predicted noises. Subtract it from img_noisy to get the predicted image.
            img_stu_pred = self.predict_start_from_noise(img_noisy, t, img_stu_pred)
        # otherwise, objective is 'pred_x0', and img_stu_pred is already the predicted image.

        if self.iter_count % 1000 == 0:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            sample_dir = f'{self.sample_dir}/single'
            os.makedirs(sample_dir, exist_ok=True)

            img_gtaug_save_path  = f'{sample_dir}/{self.iter_count}-{local_rank}-aug.png'
            unnorm_save_image(img_gt,   img_gtaug_save_path,  nrow = 8)
            #img_gtorig_save_path = f'{sample_dir}/{self.iter_count}-{local_rank}-orig.png'
            #unnorm_save_image(img_orig, img_gtorig_save_path, nrow = 8)

            #print("GT images for single-image class guidance are saved to", img_gt_save_path)
            img_noisy_save_path = f'{sample_dir}/{self.iter_count}-{local_rank}-noisy.png'
            unnorm_save_image(img_noisy, img_noisy_save_path, nrow = 8)
            #print("Noisy images for single-image class guidance are saved to", img_noisy_save_path)
            img_pred_save_path  = f'{sample_dir}/{self.iter_count}-{local_rank}-pred.png'
            unnorm_save_image(img_stu_pred, img_pred_save_path, nrow = 8)
            #print("Predicted images are saved to", img_pred_save_path)

        feat_stu = self.denoise_fn.extract_pre_feat(self.denoise_fn.consistency_feat_ext, img_stu_pred, ref_shape=None, 
                                                     has_grad=True, use_head_feat=self.consistency_use_head_feat)

        loss_cls_guide = self.consist_loss_fn(feat_stu, feat_gt)
        return loss_cls_guide

    # inject random noise into x_start. sqrt_one_minus_alphas_cumprod_t is the std of the noise.
    def q_sample(self, x_start, t, noise=None, distill_t_frac=-1):
        assert distill_t_frac <= 1

        noise = default(noise, lambda: torch.randn_like(x_start))
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
        noise = default(noise, lambda: torch.randn_like(x_start))

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

        model_output_dict    = self.denoise_fn(x, t, classes=classes, img_tea=x_tea)
        pred_stu, pred_tea   = model_output_dict['pred_stu'],   model_output_dict['pred_tea']
        noise_feat, tea_feat = model_output_dict['noise_feat'], model_output_dict['tea_feat']

        if self.objective == 'pred_noise':
            # Laplacian loss doesn't help. Instead, it makes convergence very slow.
            if self.loss_type == 'lap':
                target = x_start
                # original pred_stu is predicted noise. Subtract it from noisy x to get the predicted x_start.
                # LapLoss always compares the predicted x_start to the original x_start.
                pred_stu = self.predict_start_from_noise(x, t, pred_stu)
            else:
                # Compare groundtruth noise vs. predicted noise.
                target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss_stu = self.loss_fn(pred_stu, target)
        if self.distillation_type != 'none':
            loss_tea    = self.loss_fn(pred_tea, target)
            if self.featnet_type != 'mini':
                loss_align_tea_stu = F.l1_loss(noise_feat, tea_feat.detach())
            else:
                loss_align_tea_stu = torch.zeros_like(loss_stu)
        else:
            loss_tea = torch.tensor(0, device=x_start.device)
            loss_align_tea_stu = torch.zeros_like(loss_stu)

        if self.cls_guide_type != 'none':
            if self.cls_guide_type == 'single':
                loss_cls_guide = self.calc_cls_single_loss(x_start, x_orig, classes)
            elif self.cls_guide_type == 'interp':
                loss_cls_guide = self.calc_cls_interp_loss(x_start, x_orig, classes)
        else:
            loss_cls_guide = torch.zeros_like(loss_stu)

        loss = loss_stu + loss_tea + \
                self.align_tea_stu_feat_weight * loss_align_tea_stu + \
                self.cls_guide_loss_weight * loss_cls_guide

        # Capping loss at 1 might helps early iterations when losses are unstable (occasionally very large).
        if loss > 1:
            loss = loss / loss.item()
        return { 'loss': loss, 'loss_stu': loss_stu, 'loss_tea': loss_tea, 'loss_cls_guide': loss_cls_guide }

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
            