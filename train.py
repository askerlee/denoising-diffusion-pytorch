from model import Unet, GaussianDiffusion, EMA, Dataset, cycle, DistributedDataParallelPassthrough, \
                  sample_images, AverageMeters, print0, reduce_tensor
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
import os
import copy
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        local_rank = -1,
        world_size = 1,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 1e-4,
        weight_decay = 0,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = True,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_workers = 5,
        debug = False,
    ):
        super().__init__()
        # model: GaussianDiffusion instance.
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.local_rank = local_rank
        self.world_size = world_size
        self.ds = dataset
        self.debug = debug
        if not self.debug:
            # DistributedSampler does shuffling by default.
            sampler = DistributedSampler(self.ds)
        else:
            sampler = None

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, sampler = sampler, 
                                        pin_memory=True, drop_last=True, num_workers=num_workers),
                        sampler)
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=weight_decay)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()
        self.loss_meter = AverageMeters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        if self.local_rank > 0:
            return

        data = {
            'step':     self.step,
            'model':    self.model.state_dict(),
            'ema':      self.ema_model.state_dict(),
            'scaler':   self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        if self.local_rank < 0:
            self.model.load_state_dict(convert(data['model']))
            self.ema_model.load_state_dict(convert(data['ema']))
        else:
            self.model.load_state_dict(data['model'])
            self.ema_model.load_state_dict(data['ema'])

        self.step = data['step']
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        is_master = (self.local_rank <= 0)
        with tqdm(initial = self.step, total = self.train_num_steps, disable=not is_master) as pbar:

            while self.step < self.train_num_steps:
                # Back up the weight of interp_feat_ext.
                interp_feat_ext = copy.deepcopy(self.model.denoise_fn.interp_feat_ext)

                for i in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    img     = data['img'].cuda()
                    classes = data['classes'].cuda()

                    with autocast(enabled = self.amp):
                        loss_dict = self.model(img, classes, step=self.step)
                        # For multiple GPUs, loss is a list.
                        # Since the loss on each GPU is MAE per pixel, we should also average them here,
                        # to make the loss consistent with being on a single GPU.
                        loss = loss_dict['loss'].mean()
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    loss_stu = loss_dict['loss_stu'].mean()
                    loss_stu = reduce_tensor(loss_stu, self.world_size)
                    self.loss_meter.update('loss_stu', loss_stu.item())
                    avg_loss_stu = self.loss_meter.avg['disp']['loss_stu']
                    desc_items = [ f's {loss_stu.item():.3f}/{avg_loss_stu:.3f}' ]

                    if args.distillation_type != 'none':
                        loss_tea = loss_dict['loss_tea'].mean()
                        loss_tea = reduce_tensor(loss_tea, self.world_size)
                        self.loss_meter.update('loss_tea', loss_tea.item())
                        avg_loss_tea = self.loss_meter.avg['disp']['loss_tea']
                        desc_items.append( f't {loss_tea.item():.3f}/{avg_loss_tea:.3f}' )
                    if args.interp_loss_weight > 0:
                        loss_interp = loss_dict['loss_interp'].mean()
                        loss_interp = reduce_tensor(loss_interp, self.world_size)
                        self.loss_meter.update('loss_interp', loss_interp.item())
                        avg_loss_interp = self.loss_meter.avg['disp']['loss_interp']
                        desc_items.append( f'i {loss_interp.item():.3f}/{avg_loss_interp:.3f}' )

                    desc = ', '.join(desc_items)
                    pbar.set_description(desc)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                # Restore interp_feat_ext weights to keep it from being updated.
                self.model.denoise_fn.interp_feat_ext = interp_feat_ext

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.loss_meter.disp_reset()
                    # ema_model: GaussianDiffusion
                    self.ema_model.eval()

                    milestone = self.step // self.save_and_sample_every
                    img_save_path = str(self.results_folder / f'sample-{milestone}.png')
                    self.save(milestone)
                    sample_images(self.ema_model, 36, self.batch_size, img_save_path)               

                self.step += 1
                pbar.update(1)

        print0('Training completed')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
parser.add_argument('--bs', dest='batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--cp', type=str, dest='cp_path', default=None, help="The path of a model checkpoint")
parser.add_argument('--sample', dest='sample_only', action='store_true', help='Do sampling using a trained model')
parser.add_argument('--workers', dest='num_workers', type=int, default=5, 
                    help="Number of workers for data loading. On machines with slower disk IO, this should be higher.")
parser.add_argument('--debug', action='store_true', help='Debug the diffusion process')

parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--noamp', dest='amp', default=True, action='store_false', help='Do not use mixed precision')
parser.add_argument('--ds', type=str, default='imagenet', help="The path of training dataset")
parser.add_argument('--out', dest='results_folder', type=str, default='results', help="The path to save checkpoints and sampled images")
parser.add_argument('--times', dest='num_timesteps', type=int, default=1000, 
                    help="Number of maximum diffusion steps")
parser.add_argument('--mem', dest='memory_size', type=int, default=1024, 
                    help="Number of memory cells in each attention layer")
parser.add_argument('--sched', dest='alpha_beta_schedule', type=str, choices=['cosb', 'lina', 'linb'], default='linb', 
                    help="Type of alpha/beta schedule")

parser.add_argument('--losstype', dest='loss_type', type=str, choices=['l1', 'l2', 'lap'], default='l1', 
                    help="Type of image denoising loss")
parser.add_argument('--obj', dest='objective_type', type=str, choices=['pred_noise', 'pred_x0'], default='pred_noise', 
                    help="Type of denoising objective")
parser.add_argument('--sampinterval', dest='save_sample_interval', type=int, default=1000, 
                    help="Every N iterations, save model and sample example images")
parser.add_argument('--featnet', dest='featnet_type', 
                    choices=[ 'none', 'mini', 'resnet34', 'resnet18', 'repvgg_b0', 
                              'mobilenetv2_120d', 'vit_base_patch8_224', 'vit_tiny_patch16_224' ], 
                    default='vit_tiny_patch16_224', 
                    help='Type of the feature network. Used by the distillation and interpolation losses.')
parser.add_argument('--distill', dest='distillation_type', choices=['none', 'tfrac'], default='none', 
                    help='Distillation type')
parser.add_argument('--dtfrac', dest='distill_t_frac', default=0.8, type=float, 
                    help='Fraction of t of noise to be added to teacher images '
                         '(the smaller, the less noise is added to groundtruth images)')                          
parser.add_argument('--tuneteacher', dest='finetune_tea_feat_ext', default=False, action='store_true', 
                    help='Fine-tune the pretrained image feature extractor of the teacher model (default: freeze it).')
parser.add_argument('--alignfeat', dest='align_tea_stu_feat_weight', default=0.0, type=float, 
                    help='Align the features of the feature extractors of the teacher and the student. '
                    'Default: 0.0, meaning no alignment.')
parser.add_argument('--clsembed', dest='cls_embed_type', choices=['none', 'tea_stu', 'tea_only'], default='none', 
                    help='How class embedding is incorporated in the student and teacher')
parser.add_argument('--winterp', dest='interp_loss_weight', default=0.0, type=float, 
                    help='Interpolate random image pairs for better latent space semantics structure. '
                    'Default: 0.0, meaning no interpolation loss.')
parser.add_argument('--interpfullfeat', dest='interp_use_cls_feat', action='store_false', 
                    help='Do not use CLS feat, instead use the full featmaps when computing interpolation loss')                    

torch.set_printoptions(sci_mode=False)
args = parser.parse_args()
local_rank = int(os.environ.get('LOCAL_RANK', 0))
if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.distributed = args.world_size > 1
else:
    args.world_size = 1
args.batch_size //= args.world_size

args.local_rank = local_rank
print0(f"Args:\n{args}")

if args.distillation_type == 'tfrac' and args.featnet_type == 'none':
    print0("Distillation type is 'tfrac', but no feature network is specified. ")
    exit(0)

if args.interp_loss_weight > 0:
    if args.featnet_type == 'none':
        print0("Interpolation loss is enabled, but no feature network is specified.")
        exit(0)
    if args.featnet_type == 'repvgg_b0':
        print0("repvgg_b0 doesn't work with interpolation loss. Recommended: vit_tiny_patch16_224")
        exit(0)

if not args.debug:
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

torch.cuda.set_device(args.local_rank)
args.seed = 1234
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

dataset = Dataset(args.ds, image_size=128)
num_images = len(dataset)
num_classes = num_images

print0(f"world size: {args.world_size}, batch size per GPU: {args.batch_size}, seed: {args.seed}")

unet = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    # with_time_emb = True, do time embedding.
    memory_size = args.memory_size,
    num_classes = num_images,
    # if do distillation and featnet_type=='mini', use a miniature of original images as privileged information to train the teacher model.
    # if do distillation and featnet_type=='resnet34' or another model name, 
    # use image features extracted with a pretrained model to train the teacher model.
    featnet_type = args.featnet_type,
    distillation_type = args.distillation_type,
    # if finetune_tea_feat_ext=False,
    # do not finetune the pretrained image feature extractor of the teacher model.
    finetune_tea_feat_ext = args.finetune_tea_feat_ext,
    cls_embed_type = args.cls_embed_type,
)

diffusion = GaussianDiffusion(
    unet,                               # denoise_fn
    image_size = 128,                   # Input image is resized to image_size before augmentation.
    num_timesteps = args.num_timesteps, # number of maximum diffusion steps
    alpha_beta_schedule = args.alpha_beta_schedule, # alpha/beta schedule
    loss_type = args.loss_type,         # L1, L2, lap (Laplacian)
    objective = args.objective_type,    # objective type, pred_noise or pred_x0
    # if do distillation and featnet_type=='mini', use a miniature of original images as privileged information to train the teacher model.
    # if do distillation and featnet_type=='resnet34' or another model name, 
    # use image features extracted with a pretrained model to train the teacher model.
    featnet_type = args.featnet_type,
    distillation_type = args.distillation_type,
    distill_t_frac = args.distill_t_frac,
    cls_embed_type = args.cls_embed_type,
    num_classes = num_images,
    interp_loss_weight = args.interp_loss_weight,
    interp_use_cls_feat = args.interp_use_cls_feat,
    align_tea_stu_feat_weight = args.align_tea_stu_feat_weight,
    output_dir = args.results_folder,
    debug = args.debug,
)

diffusion.cuda()
# default using two GPUs.
diffusion = DistributedDataParallelPassthrough(diffusion, device_ids=[local_rank], 
                                               output_device=local_rank,
                                               find_unused_parameters=True)

if args.cp_path is not None:
    diffusion.load_state_dict(torch.load(args.cp_path)['ema'])

if args.sample_only:
    assert args.cp_path is not None, "Please specify a checkpoint path to load for sampling"
    cp_trunk = os.path.basename(args.cp_path).split('.')[0]
    sample_images(diffusion, 36, args.batch_size, str(args.results_folder / f'{cp_trunk}-sample.png'))
    exit()

trainer = Trainer(
    diffusion,
    dataset,                            # Dataset instance.
    local_rank = args.local_rank,       # Local rank of the process.
    world_size = args.world_size,       # Total number of processes.
    train_batch_size = args.batch_size, # default: 32
    train_lr = args.lr,                 # default: 1e-4
    train_num_steps = 700000,           # total training steps
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    amp = args.amp,                     # turn on mixed precision. Default: True
    results_folder = args.results_folder,
    save_and_sample_every = args.save_sample_interval,
)

trainer.train()
