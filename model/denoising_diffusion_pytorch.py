import math
import torch
from torch import nn, einsum
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import utils
from inspect import isfunction
from functools import partial

import timm
from einops import rearrange
from .laplacian import LapLoss
from .utils import timm_extract_features #, dual_teaching_loss

timm_model2dim = { 'resnet34': 512,
                   'resnet18': 512,
                   'repvgg_b0': 1280,
                   'mobilenetv2_120d': 1280,
                   'vit_base_patch8_224': 768,
                   'vit_tiny_patch16_224': 192 }

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 5x faster than tensor.repeat_interleave().
def repeat_interleave(x, n, dim):
    if dim < 0:
        dim += x.dim()
    x2 = x.unsqueeze(dim+1)
    repeats = [1] * (len(x.shape) + 1)
    repeats[dim+1] = n
    new_shape = list(x.shape)
    new_shape[dim] *= n
    x_rep = x2.repeat(repeats)
    x_new = x_rep.reshape(new_shape)
    return x_new

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
            nn.Linear(time_emb_dim, dim_out)
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
        learned_variance = False,
        featnet_type = 'none',
        finetune_tea_feat_ext = False,
        do_distillation = False,
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
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

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
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim, memory_size=0)))
        # Seems setting kernel size to 1 leads to slightly worse images?
        self.mid_block2 = block_klass(mid_dim, mid_dim, kernel_size=1, time_emb_dim = time_dim)

        self.featnet_type      = featnet_type
        self.finetune_tea_feat_ext  = finetune_tea_feat_ext
        self.do_distillation   = do_distillation

        if self.featnet_type != 'none' and self.featnet_type != 'mini':
            # Tried 'efficientnet_b0', but it doesn't perform well.
            # 'resnet34', 'resnet18', 'repvgg_b0'
            self.featnet_type   = self.featnet_type 
            self.featnet_dim    = timm_model2dim[self.featnet_type]
            self.dist_feat_ext_tea  = timm.create_model(self.featnet_type, pretrained=True)
            if self.do_distillation:
                self.dist_feat_ext_stu  = timm.create_model(self.featnet_type, pretrained=True)
            else:
                self.dist_feat_ext_stu  = None
        else:
            self.featnet_dim  = 0
            self.dist_feat_ext_tea = None
            self.dist_feat_ext_stu = None

        if (not self.do_distillation) or self.featnet_type == 'none':
            extra_up_dim = 0
        elif self.featnet_type == 'mini':
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
    def extract_pre_feat(self, feat_extractor, img, ref_shape, do_finetune=False):
        if self.featnet_type == 'none':
            return None

        if self.featnet_type == 'mini':
            # A miniature image as teacher's priviliged information. 
            # 128x128 images will be resized to 16x16 below.
            distill_feat = img
        else:
            if do_finetune:
                distill_feat = timm_extract_features(feat_extractor, img)
            else:
                # Wrap the feature extraction with no_grad() to save RAM.
                with torch.no_grad():
                    distill_feat = timm_extract_features(feat_extractor, img)                

        if ref_shape is not None:
            # For 128x128 images, vit features are 4x4. Resize to 16x16.
            distill_feat = F.interpolate(distill_feat, size=ref_shape, mode='bilinear', align_corners=False)
        # Otherwise, do not resize distill_feat.
        return distill_feat

    def forward(self, x, time, img_tea=None):
        init_noise = x
        x = self.init_conv(x)
        # t: time embedding.
        if exists(self.time_mlp):
            t = self.time_mlp(time.flatten())
            # t: [batch, 1, 1, time_dim=256].
            t = t.view(time.shape[0], *((1,) * (len(x.shape) - 2)), -1)
        else:
            t = None

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

        # Always fine-tune the student feature extractor.
        noise_feat = self.extract_pre_feat(self.dist_feat_ext_stu, init_noise, mid_feat.shape, 
                                           do_finetune=True)
        for ind, (block1, block2, attn, upsample) in enumerate(self.ups):
            if ind == 0:
                x = torch.cat((x, h[-ind-1], noise_feat), dim=1)
            else:
                x = torch.cat((x, h[-ind-1]), dim=1)

            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            # All blocks above don't change feature resolution. Only upsample explicitly here.
            # upsample() is a 4x4 conv, stride-2 transposed conv.
            x = upsample(x)

        pred_stu = self.final_conv(x)
        tea_feat  = None

        # img_tea is provided. Do distillation.
        if self.do_distillation and img_tea is not None:
            x = mid_feat
            # finetune_tea_feat_ext controls whether to fine-tune the teacher feature extractor.
            tea_feat = self.extract_pre_feat(self.dist_feat_ext_tea, img_tea, mid_feat.shape, 
                                             do_finetune=self.finetune_tea_feat_ext)

            for ind, (block1, block2, attn, upsample) in enumerate(self.ups_tea):
                if ind == 0:
                    x = torch.cat((x, h[-ind-1], tea_feat), dim=1)
                else:
                    x = torch.cat((x, h[-ind-1]), dim=1)

                x = block1(x, t)
                x = block2(x, t)
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
        objective = 'pred_noise',
        featnet_type = 'none',
        do_distillation = False,
        distill_t_frac = 0.,
        interp_loss_weight = 0.,
        align_tea_stu_feat_weight = 0,
        output_dir = './results',
        debug = False,
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
        self.image_size = image_size
        self.denoise_fn = denoise_fn    # Unet
        self.objective = objective
        self.featnet_type = featnet_type
        self.do_distillation = do_distillation
        self.distill_t_frac = distill_t_frac if self.do_distillation  else -1
        self.interp_loss_weight = interp_loss_weight
        self.align_tea_stu_feat_weight = align_tea_stu_feat_weight
        self.output_dir = output_dir
        self.debug = debug
        self.num_timesteps = num_timesteps
        self.alpha_beta_schedule = alpha_beta_schedule
        
        if self.alpha_beta_schedule == 'cosb':
            print("Use cosine_beta_schedule")
            betas = cosine_beta_schedule(self.num_timesteps)
        elif self.alpha_beta_schedule == 'lina':
            print("Use linear_alpha_schedule")
            betas = linear_alpha_schedule(self.num_timesteps)
        elif self.alpha_beta_schedule == 'linb':
            print("Use linear_beta_schedule")
            betas = linear_beta_schedule(self.num_timesteps)            
        else:
            breakpoint()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.loss_type = loss_type
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
    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output_dict = self.denoise_fn(x, t)
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
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    def noisy_interpolate(self, x1, x2, t_batch, w = 0.5):
        assert x1.shape == x2.shape
        # Apply the same t_batch to x1 and x2, respectively. 
        # Otherwise it's difficult to deal with different time embeddings of xt1 and xt2.
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batch), (x1, x2))

        interp_img = w * x1 + (1 - w) * x2 
        return interp_img

    @torch.no_grad()
    def image_interpolate(self, x1, x2, t = None, w = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        t_batch = torch.stack([torch.tensor(t, device=device)] * b)
        img = self.noisy_interpolate(x1, x2, t_batch, w)
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # x is already added with noises.
    def calc_interpolation_loss(self, img_noisy, img_gt, t, min_interp_w = 0.2):
        b = img_noisy.shape[0]
        assert b % 2 == 0
        b2 = b // 2
        x1, x2 = img_noisy[:b2], img_noisy[b2:]
        t2 = t[:b2]
        feat_gt = self.denoise_fn.extract_pre_feat(self.denoise_fn.dist_feat_ext_tea, img_gt, None, 
                                                   do_finetune=False)        
        feat_gt1, feat_gt2 = feat_gt[:b2], feat_gt[b2:]

        w = torch.rand((b2, ), device=img_noisy.device)
        # Normalize w into [min_interp_w, 1-min_interp_w], i.e., [0.2, 0.8].
        w = (1 - 2 * min_interp_w) * w + min_interp_w
        w = w.view(b2, *((1,) * (len(img_noisy.shape) - 1)))
        interp_img = w * x1 + (1 - w) * x2 

        # Setting the last param (img_tea) to None, so that teacher module won't be executed, 
        # to reduce unnecessary compute.
        model_output_dict = self.denoise_fn(interp_img, t2, None)
        interp_pred = model_output_dict['pred_stu']

        if self.objective == 'pred_noise':
            # interp_pred is the predicted noises. Subtract it from interp_img to get the predicted image.
            interp_pred = self.predict_start_from_noise(interp_img, t2, interp_pred)
        # otherwise, objective is 'pred_x0', and interp_pred is already the predicted image.
            
        feat_interp = self.denoise_fn.extract_pre_feat(self.denoise_fn.dist_feat_ext_tea, interp_pred, None, 
                                                       do_finetune=False)

        loss_interp1 = self.loss_fn(feat_interp, feat_gt1, reduction='none')
        loss_interp2 = self.loss_fn(feat_interp, feat_gt2, reduction='none')
        # if neighbor_mask[i, pos] = 1, i.e., this pixel's feature is more similar to sub-batch1, 
        # then use loss weight w. Otherwise, it's more similar to sub-batch2, use loss weight 1-w.
        neighbor_mask = (loss_interp1 < loss_interp2).float()
        loss_weight = neighbor_mask * w + (1 - neighbor_mask) * (1 - w)
        # The more similar features from tea_feat of either sub-batch1 or sub-batch2 are selected 
        # to compute the loss with feat_interp at each pixel.
        loss_interp = torch.minimum(loss_interp1, loss_interp2) * loss_weight
        loss_interp = loss_interp.mean()

        return loss_interp

    # inject random noise into x_start. sqrt_one_minus_alphas_cumprod_t is the std of the noise.
    def q_sample(self, x_start, t, noise=None, distill_t_frac=-1, step=0):
        assert distill_t_frac <= 1

        noise = default(noise, lambda: torch.randn_like(x_start))
        #x_start_weight = extract_tensor(self.sqrt_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)
        #noise_weight   = extract_tensor(self.sqrt_one_minus_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)
        # t serves as a tensor of indices, to extract elements from alphas_cumprod.
        alphas_cumprod  = extract_tensor(self.alphas_cumprod, t, x_start.shape)
        x_start_weight  = torch.sqrt(alphas_cumprod)
        noise_weight    = torch.sqrt(1 - alphas_cumprod)
        x_noisy1 = x_start_weight * x_start + noise_weight * noise

        if self.debug and step < 10:
            print(f'{step} x_start_weight\n{x_start_weight.flatten()}')
            print(f'{step} noise_weight\n{noise_weight.flatten()}')

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

                if self.debug and step < 10:
                    print(f'{step} x_start_weight2\n{x_start_weight2.flatten()}')
                    print(f'{step} noise_weight2\n{noise_weight2.flatten()}')

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

    # x_start: initial image.
    def p_losses(self, x_start, t, noise = None, step=0):
        b, c, h, w = x_start.shape
        # noise: a Gaussian noise for each pixel.
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, distill_t_frac=self.distill_t_frac, step=step)
        # Sample an easier x2 according to a smaller t.
        if self.distill_t_frac == -1:
            x       = x_noisy
            x_tea   = None
        else:
            x, x_tea = x_noisy

        if self.debug and step < 10:
            utils.save_image(x, f'{self.output_dir}/{step}-stu.png')
            if x_tea is not None:
                utils.save_image(x_tea, f'{self.output_dir}/{step}-tea.png')

        model_output_dict    = self.denoise_fn(x, t, x_tea)
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
        if self.do_distillation:
            loss_tea    = self.loss_fn(pred_tea, target)
            if self.featnet_type != 'mini':
                loss_align_tea_stu = F.l1_loss(noise_feat, tea_feat.detach())
            else:
                loss_align_tea_stu = 0
        else:
            loss_tea = torch.tensor(0, device=x_start.device)
            loss_align_tea_stu = 0

        if self.interp_loss_weight > 0:
            loss_interp = self.calc_interpolation_loss(x, x_start, t)
        else:
            loss_interp = 0

        loss = loss_stu + loss_tea + \
                self.align_tea_stu_feat_weight * loss_align_tea_stu + \
                self.interp_loss_weight * loss_interp

        # Capping loss at 1 might helps early iterations when losses are unstable (occasionally very large).
        if loss > 1:
            loss = loss / loss.item()
        return {'loss': loss, 'loss_stu': loss_stu, 'loss_tea': loss_tea}

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        if not (h == img_size and w == img_size):
            print(f'height and width of image must be {img_size}')
            breakpoint()
        
        if self.interp_loss_weight > 0:
            assert b % 2 == 0
            # Two sub-batches use the same random timesteps.
            t = torch.randint(0, self.num_timesteps, (b//2, ), device=device).long().repeat(2)
        else:
            # t: random numbers of steps between 0 and num_timesteps - 1 (num_timesteps default is 1000)
            # (b,): different random steps for different images in a batch.
            t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

