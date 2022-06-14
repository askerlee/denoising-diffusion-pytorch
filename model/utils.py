import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import numpy as np
import timm
from .laplacian import LapLoss
import imgaug.augmenters as iaa
from torchvision.transforms import ColorJitter, ToTensor, ToPILImage

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    
# dataset classes
class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        '''
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
        '''
        affine_prob     = 0.1
        perspect_prob   = 0.1 

        tgt_height = tgt_width = image_size
        self.geo_aug_func = iaa.Sequential(
                [
                    # Resize the image to the size (h, w). When the original image is too big, 
                    # the first resizing avoids cropping too small fractions of the whole image.
                    # For Sintel, (h, w) = (288, 680), around 2/3 of the original image size (436, 1024).
                    # For Vimeo,  (h, w) = (256, 488) is the same as the original image size.
                    # iaa.Resize({ 'height': self.h, 'width': self.w }),
                    # As tgt_width=tgt_height=224, the aspect ratio is always 1.
                    # Randomly crop to 256*256 (Vimeo) or 288*288 (Sintel).
                    iaa.CropToAspectRatio(aspect_ratio=tgt_width/tgt_height, position='uniform'),
                    # Crop a random length from uniform(0, 2*delta) (equal length at four sides). 
                    # The mean crop length is delta, and the mean size of the output image is
                    # (self.h - 2*delta) * (self.h - 2*delta) = tgt_height * tgt_height (=tgt_width).
                    iaa.Crop(percent=(0, 0.1), keep_size=False),
                    # Resize the image to the shape of target size.
                    iaa.Resize({'height': tgt_height, 'width': tgt_width}),
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # Horizontally flip 50% of all images
                    # iaa.Flipud(0.5),  # Vertically flip 50% of all images
                    # iaa.Sometimes(0.2, iaa.Rot90((1, 3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                    # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                    iaa.Sometimes(affine_prob, iaa.Affine(
                            rotate=(-20, 20), # rotate by -45 to +45 degrees
                            shear=(-16, 16), # shear by -16 to +16 degrees
                            order=1,
                            cval=(0,255),
                            mode='constant'  
                            # Previously mode='reflect' and no PerspectiveTransform => worse performance.
                            # Which is the culprit? maybe mode='reflect'? 
                            # But PerspectiveTransform should also have positive impact, as it simulates
                            # a kind of scene changes due to motion.
                    )),
                    iaa.Sometimes(perspect_prob, 
                                  iaa.PerspectiveTransform(scale=(0.01, 0.15), cval=(0,255), mode='constant')), 
                    iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades?
                    # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                    # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                    # iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),
                    # iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                ])

        self.tv_transform = transforms.Compose([
            ToPILImage(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14),
            ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = np.array(Image.open(path))
        img_aug = self.geo_aug_func.augment_image(np.array(img)).copy()
        return self.tv_transform(img_aug)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def sample_images(model, num_images, batch_size, img_save_path):
    batches = num_to_groups(num_images, batch_size)
    all_images_list = list(map(lambda n: model.sample(batch_size=n), batches))
    all_images = torch.cat(all_images_list, dim=0)
    utils.save_image(all_images, img_save_path, nrow = 6)
    print(f"Sampled {img_save_path}")

# For CNN models, just forward to forward_features().
# For ViTs, patch the original timm code to keep the spatial dimensions of the extracted image features.
# use_cls_feat: collapse geometric dimensions of the features.
def timm_extract_features(model, x, use_cls_feat=False):
    if type(model) != timm.models.vision_transformer.VisionTransformer:
        feat = model.forward_features(x)
        if use_cls_feat:
            feat = feat.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        return feat

    # img_size and patch_size are tuples.
    img_size = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

    # timm vit only accepts fixed images. So x has to be resized to [224, 224].
    if x.shape[2:] != img_size:
        x = torch.nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=False)

    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_token, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    x = model.blocks(x)
    x = model.norm(x)

    if use_cls_feat:
        x = x[:, [0]].permute(0, 2, 1).reshape(x.shape[0], -1, 1, 1)
    else:
        # Remove the 'CLS' token from the output.
        x = x[:, 1:].permute(0, 2, 1).reshape(x.shape[0], -1, *out_size)
    return x


# Dual teaching doesn't work.
def dual_teaching_loss(img_gt, img_stu, img_tea):
    loss_distill = 0
    # Ws[0]: weight of teacher -> student.
    # Ws[1]: weight of student -> teacher.
    # Two directions could take different weights.
    # Set Ws[1] to 0 to disable student -> teacher.
    Ws = [1, 0.5]
    use_lap_loss = False
    # Laplacian loss performs better in earlier epochs, but worse in later epochs.
    # Moreover, Laplacian loss is significantly slower.
    if use_lap_loss:
        loss_fun = LapLoss(max_levels=3, reduction='none')
    else:
        loss_fun = nn.L1Loss(reduction='none')

    for i in range(2):
        student_error = loss_fun(img_stu, img_gt).mean(1, True)
        teacher_error = loss_fun(img_tea, img_gt).mean(1, True)
        # distill_mask indicates where the warped images according to student's prediction 
        # is worse than that of the teacher.
        # If at some points, the warped image of the teacher is better than the student,
        # then regard the flow at these points are more accurate, and use them to teach the student.
        distill_mask = (student_error > teacher_error + 0.01).float().detach()

        # loss_distill is the sum of the distillation losses at 2 directions.
        loss_distill += Ws[i] * ((img_tea.detach() - img_stu).abs() * distill_mask).mean()

        # Swap student and teacher, and calculate the distillation loss again.
        img_stu, img_tea = img_tea, img_stu
        # The distillation loss from the student to the teacher is given a smaller weight.

    return loss_distill

class AverageMeters:
    """Computes and stores the average and current values of given keys"""
    def __init__(self):
        self.total_reset()

    def total_reset(self):
        self.val   = {'total': {}, 'disp': {}}
        self.avg   = {'total': {}, 'disp': {}}
        self.sum   = {'total': {}, 'disp': {}}
        self.count = {'total': {}, 'disp': {}}

    def disp_reset(self):
        self.val['disp']   = {}
        self.avg['disp']   = {}
        self.sum['disp']   = {}
        self.count['disp'] = {}

    def update(self, key, val, n=1, is_val_avg=True):
        if type(val) == torch.Tensor:
            val = val.item()
        if type(n) == torch.Tensor:
            n = n.item()

        if np.isnan(val):
            breakpoint()

        for sig in ('total', 'disp'):
            self.val[sig][key]    = val
            self.sum[sig].setdefault(key, 0)
            self.count[sig].setdefault(key, 0.0001)
            self.avg[sig].setdefault(key, 0)
            if is_val_avg:
                self.sum[sig][key] += val * n
            else:
                self.sum[sig][key] += val

            self.count[sig][key] += n
            self.avg[sig][key]    = self.sum[sig][key] / self.count[sig][key]
