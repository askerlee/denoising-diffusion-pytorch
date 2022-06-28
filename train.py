from model import Unet, GaussianDiffusion, UnlabeledDataset, LabeledDataset, Imagenet, \
                  EMA, cycle, DistributedDataParallelPassthrough, \
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
import re

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        local_rank = -1,
        world_size = 1,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        weight_decay = 0,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = True,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        sample_dir = 'samples',
        cp_dir = 'checkpoints',
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
        self.dataset = dataset
        self.debug = debug
        if not self.debug:
            # DistributedSampler does shuffling by default.
            sampler = DistributedSampler(self.dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.dl = cycle(data.DataLoader(self.dataset, batch_size = train_batch_size, sampler = sampler, 
                                        shuffle = shuffle, pin_memory = True, 
                                        drop_last = True, num_workers = num_workers),
                        sampler)
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=weight_decay)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.sample_dir = sample_dir
        self.cp_dir = cp_dir
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(cp_dir, exist_ok=True)

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
        torch.save(data, f'{self.cp_dir}/model-{milestone:03}.pt')

    def load(self, milestone, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        
        data = torch.load(f'{self.cp_dir}/model-{milestone:03}.pt')
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
                self.model.denoise_fn.pre_update()

                for i in range(self.gradient_accumulate_every):
                    data        = next(self.dl)
                    img         = data['img'].cuda()
                    img_orig    = data['img_orig'].cuda()
                    classes     = data['cls'].cuda()

                    with autocast(enabled = self.amp):
                        loss_dict = self.model(img, img_orig, classes, iter_count=self.step)
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
                    if args.cls_guide_loss_weight > 0:
                        loss_cls_guide = loss_dict['loss_cls_guide'].mean()
                        loss_cls_guide = reduce_tensor(loss_cls_guide, self.world_size)
                        self.loss_meter.update('loss_cls_guide', loss_cls_guide.item())
                        avg_loss_cls_guide = self.loss_meter.avg['disp']['loss_cls_guide']
                        desc_items.append( f'c {loss_cls_guide.item():.3f}/{avg_loss_cls_guide:.3f}' )

                    desc = ', '.join(desc_items)
                    pbar.set_description(desc)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.model.denoise_fn.post_update()
                
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.loss_meter.disp_reset()
                    # ema_model: GaussianDiffusion
                    self.ema_model.eval()

                    milestone = self.step // self.save_and_sample_every
                    img_save_path = f'{self.sample_dir}/{milestone:03}-sample.png'
                    nn_save_path  = f'{self.sample_dir}/{milestone:03}-nn.png'
                    self.save(milestone)
                    # self.dataset is provided for nearest neighbor imgae search.
                    sample_images(self.ema_model, 36, 36, self.dataset, img_save_path, nn_save_path)               

                self.step += 1
                pbar.update(1)

        print0('Training completed')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234, help="Random seed for initialization and training")
parser.add_argument('--sampleseed', type=int, default=5678, help="Random seed for sampling")

parser.add_argument('--lr', type=float, default=4e-4, help="Learning rate")
parser.add_argument('--bs', dest='batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--cp', type=str, dest='cp_path', default=None, help="The path of a model checkpoint")
parser.add_argument('--sample', dest='sample_only', action='store_true', help='Do sampling using a trained model')
parser.add_argument('--nogeoaug', dest='do_geo_aug', action='store_false', 
                    help='Do not do geometric augmentation to training images')
parser.add_argument('--workers', dest='num_workers', type=int, default=5, 
                    help="Number of workers for data loading. On machines with slower disk IO, this should be higher.")
parser.add_argument('--debug', action='store_true', help='Debug the diffusion process')

parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--noamp', dest='amp', default=True, action='store_false', help='Do not use mixed precision')
parser.add_argument('--ds', type=str, default='imagenet', help="The path of training dataset")
parser.add_argument('--saveimg', dest='sample_dir', type=str, default='samples', 
                    help="The path to save sampled images")
parser.add_argument('--savecp',  dest='cp_dir', type=str, default='checkpoints', 
                    help="The path to save checkpoints")

parser.add_argument('--times', dest='num_timesteps', type=int, default=1000, 
                    help="Number of maximum diffusion steps")
parser.add_argument('--mem', dest='memory_size', type=int, default=2048, 
                    help="Number of memory cells in each attention layer")
parser.add_argument('--sched', dest='alpha_beta_schedule', type=str, choices=['cosb', 'lina', 'linb'], 
                    default='lina', help="Type of alpha/beta schedule")

parser.add_argument('--losstype', dest='loss_type', type=str, choices=['l1', 'l2', 'lap'], default='l1', 
                    help="Type of image denoising loss")
parser.add_argument('--consistlosstype', dest='consist_loss_type', type=str, 
                    choices=['l1', 'cosine'], default='l1', 
                    help="Type of image feature consistency loss")
parser.add_argument('--obj', dest='objective_type', type=str, choices=['pred_noise', 'pred_x0'], default='pred_noise', 
                    help="Type of denoising objective")
parser.add_argument('--sampinterval', dest='save_sample_interval', type=int, default=1000, 
                    help="Every N iterations, save model and sample example images")
parser.add_argument('--featnet', dest='featnet_type', 
                    choices=[ 'none', 'mini', 'resnet34', 'resnet18', 'repvgg_b0', 
                              'mobilenetv2_120d', 'vit_base_patch8_224', 'vit_tiny_patch16_224' ], 
                    default='vit_tiny_patch16_224', 
                    help='Type of the feature network. Used by the distillation and interpolation losses.')
parser.add_argument('--distill', dest='distillation_type', choices=['none', 'tfrac'], default='tfrac', 
                    help='Distillation type')
parser.add_argument('--dtfrac', dest='distill_t_frac', default=0.8, type=float, 
                    help='Fraction of t of noise to be added to teacher images '
                         '(the smaller, the less noise is added to groundtruth images)')                          
parser.add_argument('--tuneteacher', dest='finetune_tea_feat_ext', default=False, action='store_true', 
                    help='Fine-tune the pretrained image feature extractor of the teacher model (default: freeze it).')
parser.add_argument('--alignfeat', dest='align_tea_stu_feat_weight', default=0.001, type=float, 
                    help='Align the features of the feature extractors of the teacher and the student. '
                    'Default: 0.0, meaning no alignment.')
parser.add_argument('--clsembed', dest='cls_embed_type', choices=['none', 'tea_stu'], default='tea_stu', 
                    help='How class embedding is incorporated in the student and teacher')
parser.add_argument('--clsguide', dest='cls_guide_type', choices=['none', 'single', 'interp'], default='single', 
                    help='The type of class guidance: none, single (one class only), '
                         'or interp (interpolation between two classes to enforce class embedding linearity)')
parser.add_argument('--wclsguide', dest='cls_guide_loss_weight', default=0.001, type=float, 
                    help='Guide denoising random images with class embedding. ')
parser.add_argument('--consfullfeat', dest='consistency_use_head_feat', action='store_false', 
                    help='Use the full feature maps when computing consistency losses (e.g., class guidance loss).')
parser.add_argument('--conssharetea', dest='consist_shares_tea_feat_ext', action='store_true', 
                    help='Use the teacher feature extractor weights for the consistency loss.')

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

if args.cls_guide_loss_weight > 0:
    if args.featnet_type == 'none':
        print0("Class guidance is enabled, but feature network is not specified.")
        exit(0)
    if args.featnet_type == 'repvgg_b0':
        print0("Class guidance is enabled, but the feature network is 'repvgg_b0'. This will lead to bad performance.")
        print0("Recommended: '--featnet vit_tiny_patch16_224'.")
        exit(0)

if args.consist_shares_tea_feat_ext:
    if args.distillation_type == 'none':
        print0("Consistency loss intends to share teacher feature extractor, but distillation is disabled "
            "(no teacher feature extractor is to be shared).")
        exit(0)
    if not args.finetune_tea_feat_ext:
        print0("Consistency loss intends to share teacher feature extractor, but teacher feature extractor is not "
               "fine-tuned (Then no point to share it).")
        exit(0)

if not args.debug:
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

torch.cuda.set_device(args.local_rank)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

if args.ds == 'imagenet':
    dataset = Imagenet(args.ds, image_size=128, split='train', do_geo_aug=args.do_geo_aug)
    save_sample_images = False
    if save_sample_images:
        dataset.training = False
        dataset.save_example("imagenet128-examples")
        exit(0)
elif args.ds == '102flowers':
    dataset = LabeledDataset(args.ds, label_file='102flowers/102flower_labels.txt', 
                             image_size=128, do_geo_aug=args.do_geo_aug)
else:    
    dataset = UnlabeledDataset(args.ds, image_size=128, do_geo_aug=args.do_geo_aug)

num_classes = dataset.get_num_classes()

print0(f"world size: {args.world_size}, batch size per GPU: {args.batch_size}, seed: {args.seed}")

unet = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    # with_time_emb = True, do time embedding.
    memory_size = args.memory_size,
    num_classes = num_classes,
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
    consist_loss_type = args.consist_loss_type,  # L1 (default), cosine.
    objective = args.objective_type,    # objective type, pred_noise or pred_x0
    # if do distillation and featnet_type=='mini', use a miniature of original images as privileged information to train the teacher model.
    # if do distillation and featnet_type=='resnet34' or another model name, 
    # use image features extracted with a pretrained model to train the teacher model.
    featnet_type = args.featnet_type,
    distillation_type = args.distillation_type,
    distill_t_frac = args.distill_t_frac,
    cls_embed_type = args.cls_embed_type,
    num_classes = num_classes,
    dataset = dataset,
    consistency_use_head_feat = args.consistency_use_head_feat,
    cls_guide_type = args.cls_guide_type,
    cls_guide_loss_weight = args.cls_guide_loss_weight,
    align_tea_stu_feat_weight = args.align_tea_stu_feat_weight,
    sample_dir = args.sample_dir,
    debug = args.debug,
    sampleseed = args.sampleseed,
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
    cp_trunk = re.match(r"model-([0-9]+)", cp_trunk).group(1)
    milestone = int(cp_trunk)
    img_save_path = f'{args.sample_dir}/{milestone:03}-sample.png'
    nn_save_path  = f'{args.sample_dir}/{milestone:03}-nn.png'
    # dataset is provided for nearest neighbor imgae search.
    sample_images(diffusion, 36, 36, dataset, img_save_path, nn_save_path)
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
    sample_dir  = args.sample_dir,
    cp_dir      = args.cp_dir,
    save_and_sample_every = args.save_sample_interval,
)

trainer.train()
