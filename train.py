from model import Unet, GaussianDiffusion, EMA, Dataset, cycle, num_to_groups
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data
from torchvision import utils
import copy
from tqdm import tqdm
from pathlib import Path
# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = True,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results'
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
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(folder, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, 
                                        pin_memory=True, num_workers=5))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step':     self.step,
            'model':    self.model.state_dict(),
            'ema':      self.ema_model.state_dict(),
            'scaler':   self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                for i in range(self.gradient_accumulate_every):
                    data = next(self.dl).cuda()

                    with autocast(enabled = self.amp):
                        loss = self.model(data)
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    pbar.set_description(f'loss: {loss.item():.4f}')

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema_model.eval()

                    milestone = self.step // self.save_and_sample_every
                    batches = num_to_groups(36, self.batch_size)
                    all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                    all_images = torch.cat(all_images_list, dim=0)
                    img_save_path = str(self.results_folder / f'sample-{milestone}.png')
                    utils.save_image(all_images, img_save_path, nrow = 6)
                    self.save(milestone)
                    print(f"Sampled {img_save_path}")

                self.step += 1
                pbar.update(1)

        print('training complete')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--bs', type=int, default=32, help="Batch size")
parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
parser.add_argument('--amp', default=True, action='store_false', help='Do not use mixed precision')
parser.add_argument('--ds', type=str, default='imagenet', help="The path of training dataset")
parser.add_argument('--results_folder', type=str, default='results', help="The path to save checkpoints and sampled images")
parser.add_argument('--timesteps', type=int, default=1000, help="Number of maximum diffusion steps")
parser.add_argument('--losstype', dest='loss_type', type=str, choices=['l1', 'l2', 'lap'], default='l1', 
                    help="Type of image denoising loss")
parser.add_argument('--obj', dest='objective_type', type=str, choices=['pred_noise', 'pred_x0'], default='pred_noise', 
                    help="Type of denoising objective")

args = parser.parse_args()
print(f"Args: \n{args}")

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
    # with_time_emb = True, do time embedding.
)

# default using two GPUs.
model = nn.DataParallel(model, device_ids=args.gpus)
model.cuda()

diffusion = GaussianDiffusion(
    model,                          # denoise_fn
    image_size = 128,               # Input image is resized to image_size before augmentation.
    timesteps = args.timesteps,     # number of maximum diffusion steps
    loss_type = args.loss_type,     # lap (Laplacian), L1 or L2
    objective=args.objective_type,  # objective type, pred_noise or pred_x0
).cuda()

trainer = Trainer(
    diffusion,
    args.ds,                          # dataset path. Default: imagenet
    train_batch_size = args.bs,       # default: 32
    train_lr = args.lr,               # default: 1e-4
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = args.amp,                   # turn on mixed precision. Default: True
    results_folder = args.results_folder
)

trainer.train()
