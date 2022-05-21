from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help="Batch size")
parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--amp', default=True, action='store_false', help='Do not use mixed precision')
parser.add_argument('--ds', type=str, default='imagenet', help="The path of training dataset")
parser.add_argument('--results_folder', type=str, default='results', help="The path to save checkpoints and sampled images")
parser.add_argument('--timesteps', type=int, default=1000, help="Number of maximum diffusion steps")

args = parser.parse_args()

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
    # with_time_emb = True, do time embedding.
)

model = nn.DataParallel(model, device_ids=args.gpus)
model.cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,               # Input image is resized to image_size before augmentation.
    timesteps = args.timesteps,     # number of maximum diffusion steps
    loss_type = 'l1'                # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    args.ds,                          # dataset path. Default: imagenet
    train_batch_size = args.bs,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = args.amp,                   # turn on mixed precision. Default: True
    results_folder = args.results_folder
)

trainer.train()
