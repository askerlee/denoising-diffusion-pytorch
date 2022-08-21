from model import Unet, GaussianDiffusion, EMA, \
                  cycle, DistributedDataParallelPassthrough, \
                  AverageMeters, print0, reduce_tensor, create_training_dataset_sampler, \
                  sample_images, translate_images
import argparse
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
# from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
import os
import copy
from tqdm.auto import tqdm
import random
import numpy as np
import re
from datetime import datetime

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        data_sampler,
        local_rank = -1,
        world_size = 1,
        adam_betas = (0.9, 0.99),
        train_batch_size = 32,
        train_lr = 1e-4,
        weight_decay = 0,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        grad_clip = -1,
        amp = True,
        num_workers = 3,
        ema_update_after_step = 2000,
        ema_update_every = 10,
        ema_decay = 0.995,
        save_and_sample_every = 1000,
        sample_dir = 'samples',
        sample_seed = 5678,
        cp_dir = 'checkpoints',
        debug = False,
    ):
        super().__init__()
        # model: GaussianDiffusion instance.
        self.model = diffusion_model
        self.local_rank = local_rank
        self.is_master = (self.local_rank <= 0)

        if self.is_master:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every, 
                        update_after_step=ema_update_after_step)

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.grad_clip = grad_clip
        self.train_num_steps = train_num_steps

        self.world_size = world_size
        self.dataset = dataset
        self.debug = debug
        if not self.debug:
            # DistributedSampler does shuffling by default. So no need to shuffle in the dataloader.
            shuffle = False
        else:
            shuffle = True

        self.dl = cycle(data.DataLoader(self.dataset, batch_size = train_batch_size, sampler = data_sampler, 
                                        shuffle = shuffle, pin_memory = True, 
                                        drop_last = True, num_workers = num_workers),
                        data_sampler)
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=weight_decay, betas=adam_betas)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.sample_dir = sample_dir
        self.cp_dir = cp_dir

        if self.is_master:
            os.makedirs(sample_dir, exist_ok=True)
            os.makedirs(cp_dir, exist_ok=True)
            print(f"Saving samples to '{sample_dir}'")

        self.loss_meter = AverageMeters()

        # Sampling uses a random generator independent from training, 
        # to facililtate comparison of different methods in terms of generation quality.
        if self.is_master:
            self.sample_rand_generator = torch.Generator(device='cuda')
            self.sample_rand_generator.manual_seed(sample_seed)

    def save(self, milestone):
        if self.local_rank > 0:
            return

        data = {
            'step':     self.step,
            'model':    self.model.state_dict(),
            'ema':      self.ema.state_dict(),
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
            self.ema.load_state_dict(convert(data['ema']))
        else:
            self.model.load_state_dict(data['model'])
            self.ema.load_state_dict(data['ema'])

        self.step = data['step']
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        with tqdm(initial = self.step, total = self.train_num_steps, disable=not self.is_master) as pbar:

            while self.step < self.train_num_steps:
                self.model.denoise_fn.pre_update()

                for i in range(self.gradient_accumulate_every):
                    data        = next(self.dl)
                    img         = data['img'].cuda()
                    img_orig    = data['img_orig'].cuda()
                    classes     = data['class'].cuda()

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

                    if args.distillation_type != 'none':
                        loss_tea = loss_dict['loss_tea'].mean()
                        loss_tea = reduce_tensor(loss_tea, self.world_size)
                        self.loss_meter.update('loss_tea', loss_tea.item())
                    if args.denoise1_cls_sem_loss_weight > 0:
                        loss_cls_sem = loss_dict['loss_cls_sem'].mean()
                        loss_cls_sem = reduce_tensor(loss_cls_sem, self.world_size)
                        self.loss_meter.update('loss_cls_sem', loss_cls_sem.item())

                avg_loss_stu = self.loss_meter.avg['disp']['loss_stu']
                desc_items = [ f's {loss_stu.item():.3f}/{avg_loss_stu:.3f}' ]
                if args.distillation_type != 'none':
                    avg_loss_tea = self.loss_meter.avg['disp']['loss_tea']
                    desc_items.append( f't {loss_tea.item():.3f}/{avg_loss_tea:.3f}' )
                if args.denoise1_cls_sem_loss_weight > 0:
                    avg_loss_cls_sem = self.loss_meter.avg['disp']['loss_cls_sem']
                    desc_items.append( f'c {loss_cls_sem.item():.3f}/{avg_loss_cls_sem:.3f}' )

                desc = ', '.join(desc_items)
                pbar.set_description(desc)

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.model.denoise_fn.post_update()
                
                if self.is_master:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.loss_meter.disp_reset()
                        # ema: GaussianDiffusion
                        self.ema.eval()

                        milestone = self.step // self.save_and_sample_every
                        img_save_path = f'{self.sample_dir}/{milestone:03}-sample.png'
                        nn_save_path  = f'{self.sample_dir}/{milestone:03}-nn.png'
                        self.save(milestone)
                        # self.dataset is provided for nearest neighbor imgae search.
                        sample_images(self.ema.ema_model, self.sample_rand_generator, 36, 36, 
                                      self.dataset, img_save_path, nn_save_path)               

                self.step += 1
                pbar.update(1)

        print0('Training completed')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234, help="Random seed for initialization and training")
parser.add_argument('--sampleseed', dest='sample_seed', type=int, default=5678, 
                    help="Random seed for sampling")

parser.add_argument('--lr', type=float, default=4e-4, help="Learning rate")
parser.add_argument('--bs', dest='batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--clip', dest='grad_clip', type=float, default=-1, help="Gradient clipping")
parser.add_argument('--cp', type=str, dest='cp_path', default=None, help="The path of a model checkpoint")
parser.add_argument('--sample', dest='sample_only', action='store_true', help='Do sampling using a trained model')

parser.add_argument('--trans', dest='translate_only', action='store_true', 
                    help='Do domain translation using a trained model')
parser.add_argument('--sourceclass', dest='trans_source_class', type=int, default=-1, 
                    help='Source class of domain translation')                    
parser.add_argument('--targetclass', dest='trans_target_class', type=int, default=-1, 
                    help='Target class of domain translation')
parser.add_argument('--transnoisetfrac', dest='trans_noise_t_frac', type=float, default=0.1,
                    help='Amount of added noise used by domain translation, specified as a fraction of the max steps')
parser.add_argument('--transdenoisetfrac', dest='trans_denoise_t_frac', type=float, default=0.4,
                    help='Number of denoising steps used by domain translation, specified as a fraction of the max steps')
parser.add_argument('--transbatches', dest='trans_num_batches', type=int, default=-1,
                    help='Number of batches of images to translate (default: -1, all batches)')

parser.add_argument('--size', dest='image_size', type=int, default=128, help="Input and output image size")
parser.add_argument('--nogeoaug', dest='do_geo_aug', action='store_false', 
                    help='Do not do geometric augmentation on training images')
parser.add_argument('--nocoloraug', dest='do_color_aug', action='store_false', 
                    help='Do not do color augmentation on training images')                    
parser.add_argument('--workers', dest='num_workers', type=int, default=3, 
                    help="Number of workers for data loading. On machines with slower disk IO, this should be higher.")
parser.add_argument('--debug', action='store_true', help='Debug the diffusion process')

parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--noamp', dest='amp', default=True, action='store_false', help='Do not use mixed precision')
parser.add_argument('--ds', type=str, default='imagenet', help="The path of training dataset")
parser.add_argument('--multidom', dest='on_multi_domain', action='store_true', help='Train on multiple domains')
parser.add_argument('--saveimg', dest='sample_dir', type=str, default='samples', 
                    help="The path to save sampled images")
parser.add_argument('--savecp',  dest='cp_dir', type=str, default='checkpoints', 
                    help="The path to save checkpoints")

parser.add_argument('--steps', dest='num_timesteps', type=int, default=1000, 
                    help="Number of maximum diffusion steps")
parser.add_argument('--sampsteps', dest='sampling_timesteps', type=int, default=None, 
                    help="Number of sampling steps")

parser.add_argument('--mem', dest='memory_size', type=int, default=2048, 
                    help="Number of memory cells in each attention layer")
parser.add_argument('--sched', dest='alpha_beta_schedule', type=str, choices=['cosb', 'powa', 'linb'], 
                    default='powa', help="Type of alpha/beta schedule")
parser.add_argument('--powa-exp', dest='powa_exponent', type=float, default=3.,
                    help="Exponent of power-alpha schedule")

parser.add_argument('--losstype', dest='loss_type', type=str, choices=['l1', 'l2'], default='l1', 
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
                    help='Type of the feature network. Used by feature distillation.')

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
parser.add_argument('--clssem-featnet', dest='cls_sem_featnet_type', 
                    choices=[ 'none', 'resnet34', 'resnet18', 'repvgg_b0', 
                              'mobilenetv2_120d', 'vit_base_patch8_224', 'vit_tiny_patch16_224' ], 
                    default='vit_tiny_patch16_224', 
                    help='Type of the feature network for the class semantics loss.')
parser.add_argument('--clssem-losstype', dest='denoise1_cls_sem_loss_type', choices=['none', 'single', 'interp'], default='single', 
                    help='The type of class semantics loss: none, single (one class only), '
                         'or interp (interpolation between two classes to enforce class embedding linearity). '
                         'Even if the loss is none, class embeddings are still learned as long as num_classes >= 1.')
parser.add_argument('--wclssem', dest='denoise1_cls_sem_loss_weight', default=0.001, type=float, 
                    help='Weight of the class semantics loss that regularizes generation from noisy images with class embeddings. ')
parser.add_argument('--clssem-mintfrac', dest='denoise1_cls_sem_min_t_percentile', default=0.8, type=float,
                    help='Minimal noise amount, specified as a fraction of max time steps, to be added to images for class semantics loss.')
parser.add_argument('--clssem-use-headfeat', dest='denoise1_cls_sem_loss_use_head_feat', action='store_true', 
                    help='Use the collapsed feature maps when computing consistency losses (e.g., class semantics loss).')
parser.add_argument('--clssem-shareteafeat', dest='cls_sem_shares_tea_feat_ext', action='store_true', 
                    help='Use the teacher feature extractor weights for the consistency loss.')
parser.add_argument('--selfcond', dest='self_condition', action='store_true', 
                    help='Use self-conditioning for lower FID.')

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

if args.denoise1_cls_sem_loss_weight > 0:
    if args.featnet_type == 'none':
        print0("Class semantics loss is enabled, but feature network is not specified.")
        exit(0)
    if args.featnet_type == 'repvgg_b0':
        print0("Class semantics loss is enabled, but the feature network is 'repvgg_b0'. This will lead to bad performance.")
        print0("Recommended: '--featnet vit_tiny_patch16_224'.")
        exit(0)

if args.cls_sem_shares_tea_feat_ext:
    if args.distillation_type == 'none':
        print0("Consistency loss intends to share teacher feature extractor, but distillation is disabled "
            "(no teacher feature extractor is to be shared).")
        exit(0)
    if not args.finetune_tea_feat_ext:
        print0("Consistency loss intends to share teacher feature extractor, but teacher feature extractor is not "
               "fine-tuned (Then no point to share it).")
        exit(0)
    if args.cls_sem_featnet_type != args.featnet_type:
        print0("Consistency loss intends to share teacher feature extractor, but the feature network is not the same "
               "as the teacher feature extractor.")
        exit(0)

if not args.debug:
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

torch.cuda.set_device(args.local_rank)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

dataset, data_sampler = create_training_dataset_sampler(args)
num_classes = dataset.get_num_classes()

print0(f"world size: {args.world_size}, batch size per GPU: {args.batch_size}, seed: {args.seed}")

timestamp = datetime.now().strftime("%m%d%H%M")
if not args.translate_only:
    args.sample_dir = os.path.join(args.sample_dir, timestamp)

unet = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    # with_time_emb = True, do time embedding.
    memory_size = args.memory_size,
    self_condition = args.self_condition,
    num_classes = num_classes,
    # if do distillation and featnet_type=='mini', use a miniature of original images as privileged information to train the teacher model.
    # if do distillation and featnet_type=='resnet34' or another model name, 
    # use image features extracted with a pretrained model to train the teacher model.
    featnet_type = args.featnet_type,
    cls_sem_featnet_type = args.cls_sem_featnet_type,
    distillation_type = args.distillation_type,
    # if finetune_tea_feat_ext=False,
    # do not finetune the pretrained image feature extractor of the teacher model.
    finetune_tea_feat_ext = args.finetune_tea_feat_ext,
)

diffusion = GaussianDiffusion(
    unet,                                           # denoise_fn
    image_size = args.image_size,                   # Input image is resized to image_size before augmentation.
    num_timesteps = args.num_timesteps,             # number of maximum diffusion steps
    sampling_timesteps = args.sampling_timesteps,   # number of steps to sample from the diffusion process
    alpha_beta_schedule = args.alpha_beta_schedule, # alpha/beta schedule
    powa_exponent = args.powa_exponent,             # exponent of the power-alpha schedule.
    loss_type = args.loss_type,                     # L1, L2, lap (Laplacian)
    consist_loss_type = args.consist_loss_type,     # L1 (default), cosine.
    objective = args.objective_type,                # objective type, pred_noise or pred_x0
    # if do distillation and featnet_type=='mini', use a miniature of original images as privileged information to train the teacher model.
    # if do distillation and featnet_type=='resnet34' or another model name, 
    # use image features extracted with a pretrained model to train the teacher model.
    featnet_type = args.featnet_type,               # type of feature extractor for feature distillation.
    distillation_type = args.distillation_type,     # type of feature distillation. 'none' or 'tfrac'
    distill_t_frac = args.distill_t_frac,           # fraction of t steps to use for feature distillation.
    dataset = dataset,                              # dataset
    # use the head features of the feature extractor to compute the class semantics loss.
    denoise1_cls_sem_loss_use_head_feat = args.denoise1_cls_sem_loss_use_head_feat,
    # 'none', 'single', or 'interp'. If 'single', use the single-class semantics loss.
    # If 'interp', use the interpolated semantics loss between two classes.
    denoise1_cls_sem_loss_type = args.denoise1_cls_sem_loss_type,
    denoise1_cls_sem_loss_weight = args.denoise1_cls_sem_loss_weight,
    denoise1_cls_sem_min_t_percentile = args.denoise1_cls_sem_min_t_percentile,
    # weight of the teacher/student feature consistency loss.
    align_tea_stu_feat_weight = args.align_tea_stu_feat_weight,
    sample_dir = args.sample_dir,                   # directory to save samples.
    debug = args.debug,                             # debug mode or not.
)

diffusion.cuda()
# default using two GPUs.
diffusion = DistributedDataParallelPassthrough(diffusion, device_ids=[local_rank], 
                                               output_device=local_rank,
                                               find_unused_parameters=True)

if args.cp_path is not None:
    ema = EMA(diffusion)
    ema.load_state_dict(torch.load(args.cp_path)['ema'])
    diffusion = ema.ema_model
    print0("Loaded checkpoint from {}".format(args.cp_path))

if args.sample_only:
    assert args.cp_path is not None, "Please specify a checkpoint path to load for sampling"
    cp_trunk = os.path.basename(args.cp_path).split('.')[0]
    cp_trunk = re.match(r"model-([0-9]+)", cp_trunk).group(1)
    milestone = int(cp_trunk)
    img_save_path = f'{args.sample_dir}/{milestone:03}-sample.png'
    nn_save_path  = f'{args.sample_dir}/{milestone:03}-nn.png'
    sample_rand_generator = torch.Generator(device='cuda')
    sample_rand_generator.manual_seed(args.sample_seed)

    # dataset is provided for nearest neighbor image search.
    sample_images(diffusion, sample_rand_generator, 36, 36, dataset, img_save_path, nn_save_path)
    exit()

if args.translate_only:
    assert args.cp_path is not None, "Please specify a checkpoint path to load for domain translation"
    subfolder = "{}-{}-t{:.2f}-{}".format(dataset.class_names[args.trans_source_class], 
                                          dataset.class_names[args.trans_target_class], 
                                          args.trans_noise_t_frac, args.trans_denoise_t_frac, timestamp)
    args.sample_dir = os.path.join(args.sample_dir, subfolder)
    os.makedirs(args.sample_dir, exist_ok=True)
    print0(f"Saving translated images to {args.sample_dir}")
    trans_rand_generator = torch.Generator(device='cuda')
    trans_rand_generator.manual_seed(args.sample_seed)
    translate_images(diffusion, dataset, args.batch_size, args.trans_num_batches, 
                     args.trans_source_class, args.trans_target_class, 
                     args.sample_dir, args.trans_noise_t_frac, args.trans_denoise_t_frac, 
                     trans_rand_generator)
    exit()

trainer = Trainer(
    diffusion,
    dataset,                            # Dataset instance.
    data_sampler,                       # Dataset sampler instance.
    local_rank = args.local_rank,       # Local rank of the process.
    world_size = args.world_size,       # Total number of processes.
    train_batch_size = args.batch_size, # default: 32
    train_lr = args.lr,                 # default: 4e-4
    train_num_steps = 700000,           # total training steps.
    gradient_accumulate_every = 2,      # gradient accumulation steps
    grad_clip = args.grad_clip,         # default: -1, disabled.
    ema_decay = 0.995,                  # exponential moving average decay
    amp = args.amp,                     # turn on mixed precision. Default: True
    num_workers = args.num_workers,     # number of workers for data loading. Default: 3
    save_and_sample_every = args.save_sample_interval,  # save checkpoint and sample every this many steps. Default: 1000
    sample_dir  = args.sample_dir,      # directory to save samples.
    sample_seed = args.sample_seed,     # random seed for sampling, to enable comparison across different runs.
    cp_dir      = args.cp_dir,          # directory to save model checkpoints.
)

trainer.train()
