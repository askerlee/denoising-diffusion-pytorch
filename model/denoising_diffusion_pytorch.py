import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from einops import rearrange
from .laplacian import LapLoss

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

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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

# building block modules
# No downsampling is done in Block, i.e., it outputs a tensor of the same (H,W) as the input.
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding = 1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

# Different ResnetBlock's have different mlp (time embedding module).
# No downsampling is done in ResnetBlock.
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),      # Sigmoid Linear Unit, aka swish. https://pytorch.org/docs/stable/_images/SiLU.png
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        # block1 and block2 use SiLU activation and group norm.
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # time embedding is incorporated between block1 and block2.
    def forward(self, x, time_emb = None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb).permute(0, 3, 1, 2)
            if time_emb.shape[2] > 1 and time_emb.shape[3] > 1:
                time_emb = time_emb.repeat_interleave(h.shape[2] // time_emb.shape[2], 2).repeat_interleave(h.shape[3] // time_emb.shape[3], 3)

            h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(x)

# LinearAttention doesn't compute Q*K similarity (with softmax). 
# Instead, it computes K*V as "context" (after K is softmax-ed).
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
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
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# Ordinary attention, without softmax.
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
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
        learned_variance = False,
        do_distill = False,
    ):
        super().__init__()

        # number of input channels
        self.channels = channels

        # dim = 64, init_dim -> 42.
        init_dim = default(init_dim, dim // 3 * 2)
        # image size is the same after init_conv, as default stride=1.
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        # init_conv + 4 layers.
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 4 pairs of layers in the encoder/decoder.
        in_out = list(zip(dims[:-1], dims[1:]))

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

        num_resolutions = len(in_out)

        # Using vanilla attention in encoder/decoder takes 13x RAM, and is 3x slower.
        use_linear_attn = True
        EncDecAttention = LinearAttention if use_linear_attn else Attention

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                # A block_klass is a two-layer conv with residual connection. 
                # block_klass doesn't downsample features.
                block_klass(dim_in,  dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                # att(norm(x)) + x.
                Residual(PreNorm(dim_out, EncDecAttention(dim_out))),
                # downsampling is done with a 4x4 kernel, stride-2 conv.
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, EncDecAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.do_distill = do_distill

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            # miniature image as teacher's priviliged information.
            extra_in_dim = 3 if ind == 0 else 0

            self.ups_tea.append(nn.ModuleList([
                block_klass(dim_out * 2 + extra_in_dim, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, EncDecAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, self.out_dim, 1)
        )

    def forward(self, x, time, img_gt=None):
        x = self.init_conv(x)
        # t: time embedding.
        if exists(self.time_mlp):
            t = self.time_mlp(time.flatten())
            # When do training, time is 3-d [batch, h_grid, w_grid].
            if time.ndim == 3:
                # t: [batch, h_grid, w_grid, time_dim=256].
                t = t.view(*(time.shape), -1)
            # When do sampling, time is 1-d [batch].
            else:
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
        x = self.mid_block2(x, t)

        mid_feat = x

        for ind, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h[-ind-1]), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            # All blocks above don't change feature resolution. Only upsample explicitly here.
            # upsample() is a 4x4 conv, stride-2 transposed conv.
            x = upsample(x)

        pred_stu = self.final_conv(x)

        if self.do_distill and img_gt is not None:
            x = mid_feat
            # A miniature image as teacher's priviliged information.
            img_gt = F.interpolate(img_gt, size=x.shape[2:], mode='bilinear', align_corners=False)

            for ind, (block1, block2, attn, upsample) in enumerate(self.ups_tea):
                if ind == 0:
                    x = torch.cat((x, h[-ind-1], img_gt), dim=1)
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

        return pred_stu, pred_tea

# gaussian diffusion trainer class
# suppose t is a 1-d tensor of indices.
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # out: [b, 1, 1, 1]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        noise_grid_num = 1
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
        self.noise_grid_num = noise_grid_num

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
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
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output, _ = self.denoise_fn(x, t)

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

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # inject random noise into x_start. sqrt_one_minus_alphas_cumprod_t is the std of the noise.
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t = t.unsqueeze(1)
        # t serves as a tensor of indices, to extract elements from sqrt_alphas_cumprod.
        x_start_weight = extract(self.sqrt_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)
        noise_weight   = extract(self.sqrt_one_minus_alphas_cumprod, t.flatten(), x_start.shape).reshape(t.shape)

        if t.shape[2] > 1 or t.shape[3] > 1:
            # repeat noise weights and x_start weights to have the same size as x_start.
            x_start_weight = x_start_weight.repeat_interleave(repeats = x_start.shape[2] // t.shape[2], dim = 2).repeat_interleave(repeats = x_start.shape[3] // t.shape[3], dim = 3)
            noise_weight   = noise_weight.repeat_interleave(repeats = x_start.shape[2] // t.shape[2], dim = 2).repeat_interleave(repeats = x_start.shape[3] // t.shape[3], dim = 3)
        # if t has shape [1, 1], the it will be broadcasted to [b, c, h, w]. No need to repeat.

        return x_start_weight * x_start + noise_weight * noise

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
    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        # noise: a Gaussian noise for each pixel.
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out, _ = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            # Laplacian loss doesn't help. Instead, it makes convergence very slow.
            if self.loss_type == 'lap':
                target = x_start
                # original model_out is predicted noise. Subtract it from noisy x to get the predicted x_start.
                # LapLoss always compares the predicted x_start to the original x_start.
                model_out = self.predict_start_from_noise(x, t, model_out)
            else:
                # Compare groundtruth noise vs. predicted noise.
                target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target)
        # Capping loss at 1 might helps early iterations when losses are unstable (occasionally very large).
        if loss > 1:
            loss = loss / loss.item()
        return loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # t: random numbers of steps between 0 and num_timesteps - 1 (num_timesteps default is 1000)
        # (b,): different random steps for different images in a batch.
        t = torch.randint(0, self.num_timesteps, (b, self.noise_grid_num, self.noise_grid_num), 
                          device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

