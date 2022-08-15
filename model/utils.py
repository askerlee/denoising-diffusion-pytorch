import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms, utils
import torch.distributed as dist

from pathlib import Path
from PIL import Image
import os
import numpy as np
import timm
from .laplacian import LapLoss
import imgaug.augmenters as iaa
from torchvision.transforms import ColorJitter, ToTensor, ToPILImage, Resize
import glob
from itertools import chain

# Only print on GPU0. Avoid duplicate messages.
def print0(*print_args, **kwargs):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        print(*print_args, **kwargs)

def cycle(dl, sampler):
    epoch = 0
    while True:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for data in dl:
            yield data


def exists(x):
    return x is not None

def exists_add(x, a):
    if exists(x):
        return x + a
    else:
        return a

def fetch_attr(model, attr_name):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.__dict__[attr_name]
    else:
        return model.__dict__[attr_name]

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_tensor(tensor, world_size):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    gathered_tensor = torch.cat(tensor_list, dim=0)
    return gathered_tensor

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    
# A simple dataset class.
class BaseDataset(data.Dataset):
    def __init__(self, root, image_size, exts = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG'], 
                 do_geo_aug=True, do_color_aug=True, training=True):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.training = training
        self.paths = []
        self.cls2indices = []
        self.index2cls = []

        self.do_geo_aug = do_geo_aug
        if self.do_geo_aug:
            affine_prob     = 0.1
            perspect_prob   = 0.1 
        else:
            # Disable Affine and PerspectiveTransform. So just random cropping and resizing.
            affine_prob     = 0.0
            perspect_prob   = 0.0
        
        self.do_color_aug = do_color_aug
        if self.do_color_aug:
            gamma_prob = 0.3
        else:
            gamma_prob = 0.0

        tgt_height = tgt_width = image_size
        self.geo_aug_func = iaa.Sequential(
                [
                    # Crop the image to the target aspect ratio. Then random crop off border pixels. 
                    # Then resize to the target size.
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
                    iaa.Sometimes(gamma_prob, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades?
                ])

        self.test_transform = transforms.Compose([
            ToPILImage(),
            Resize((image_size, image_size)),
            ToTensor()
        ])

        if self.do_color_aug:
            self.tv_transform = transforms.Compose([
                ToPILImage(),
                Resize((image_size, image_size)),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14),
                ToTensor()
            ])
        else:
            self.tv_transform = self.test_transform     # No color jitter.

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = np.array(Image.open(path).convert('RGB'))
        # img_orig is a tensor of (image_size, image_size). 
        # We don't take care of the aspect ratio, as img_orig is only used for visualization.
        img_orig = self.test_transform(img)

        if self.training:
            # copy() to make the numpy array continuous.
            img_aug = self.geo_aug_func.augment_image(img).copy()
            img_aug = self.tv_transform(img_aug)
        else:
            img_aug = img_orig

        cls = self.index2cls[index]
        # print(img_aug.shape, path)
        # For small datasets such as pokemon, use index as the image classes.
        return { 'img': img_aug, 'img_orig': img_orig, 'cls': cls }

    def get_num_classes(self):
        return len(self.cls2indices)

    def sample_by_labels(self, label_list):
        '''
        Sample images that are of the classes as in label_list.
        args:
            label_list: list[int]. For imagenet, 0 ~ 999.
        return: 
            images: a dict that contains a list of images and a corresponding list of classes.
        '''
        images = { 'img': [], 'cls': [], 'img_orig': [] }
        for i in label_list:
            num_image = len(self.cls2indices[i])
            idx = np.random.randint(num_image)
            img_idx = self.cls2indices[i][idx]
            image_dict = self.__getitem__(img_idx)
            images['img'].append(image_dict['img'])
            images['cls'].append(image_dict['cls'])
            images['img_orig'].append(image_dict['img_orig'])

        images['img'] = torch.stack(images['img'])
        images['img_orig'] = torch.stack(images['img_orig'])
        images['cls'] = torch.tensor(images['cls'])

        return images

def aspect_preserving_resize(image, min_size=128):
    """
    Resize image with perserved aspect and limited min size.
    args:
        image: np.array(H, W, C) or np.array(H, W)
    """
    height, width = image.shape[:2]
    scale = min(height / min_size, width / min_size)
    new_height, new_width = int(height / scale), int(width / scale)
    assert new_height >= min_size and new_width >= min_size
    
    img = Image.fromarray(image)
    resized_image = img.resize((new_width, new_height))
    return np.array(resized_image)

# Each image is treated as a singleton class.
class SingletonDataset(BaseDataset):
    def __init__(self, root, image_size, exts = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG'], 
                 do_geo_aug=True, do_color_aug=True, training=True):
        super(SingletonDataset, self).__init__(root, image_size, exts, do_geo_aug, do_color_aug, training)

        self.paths = [ p for ext in exts for p in sorted(Path(f'{root}').glob(f'**/*.{ext}')) ]
        # Each class maps to a list of image indices. Since for simple datasets like pokemon,
        # each class is one image, so each list of indices just maps to 
        # one element: the class index itself (but note it's in a list).
        # For other datasets like Imagenet, each cls2indices entry should contain more images.
        self.cls2indices = [ [i] for i in range(len(self.paths)) ]
        # index2cls: one-to-one mapping.
        self.index2cls   = [ i for i in range(len(self.paths)) ]
        print0("Found {} images in {}".format(len(self.paths), root))

# Class labels are provided in a txt file.
class TxtLabeledDataset(BaseDataset):
    def __init__(self, root, label_file, image_size, exts = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG'], 
                 do_geo_aug=True, do_color_aug=True, training=True):
        super(TxtLabeledDataset, self).__init__(root, image_size, exts, do_geo_aug, do_color_aug, training)

        with open(label_file, 'r') as f:
            lines = f.readlines()
            # ** We assume the labels are sorted by image filenames as in self.paths. **
            # index2cls: one-to-one mapping.
            index2cls = [ int(line.strip()) for line in lines ]

        self.paths = [ p for ext in exts for p in sorted(Path(f'{root}').glob(f'**/*.{ext}')) ]
        # Each class maps to a list of image indices. However, the original labels may not be
        # named as 0, ..., n-1. For example, 102flowers has labels 1, ..., 102.
        # So we need to map the original labels to 0, ..., n-1.
        cls2indices = {}
        for k, v in enumerate(index2cls):
            cls2indices[v] = cls2indices.get(v, []) + [k]

        # Map the original labels to 0, ..., n-1.
        cls_mapping = { v: k for k, v in enumerate(sorted(cls2indices.keys())) }
        self.cls2indices = { cls_mapping[k]: v for k, v in sorted(cls2indices.items()) }
        self.index2cls = [ cls_mapping[k] for k in index2cls ]
        print0("Found {} images in {}".format(len(self.paths), root))

class Imagenet(BaseDataset):
    def __init__(self, root='imagenet128', image_size=128, split = 'train', \
        exts = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG'], do_geo_aug=True, do_color_aug=True, training=True):
        super(Imagenet, self).__init__(root, image_size, exts, do_geo_aug, do_color_aug, training)

        self.root = root
        self.split = split
        self.map_file = os.path.join(self.root, 'map_clsloc.txt')
        assert os.path.exists(self.map_file), f'Lable mapping file not found at {self.map_file}!'
        self.folder_names, self.cls2label = self.get_folder_label_mapping()
        # folder_img_list is a list of lists. Each sub-list is images in a folder (class).
        if self.split == 'test':
            self.folder_img_list = [ [ p for ext in exts \
                for p in sorted(glob.glob(os.path.join(self.root, self.split) + f'/*.{ext}')) ] ]
        else: # 'train', 'val'
            self.folder_img_list = [ [ p for ext in exts \
                # for p in Path(f'{os.path.join(self.root, self.split, fn)}').glob(f'**/*.{ext}')]\
                for p in sorted(glob.glob(os.path.join(self.root, self.split, folder_name) + f'/*.{ext}')) ] \
                for folder_name in self.folder_names ]

        self.paths = list(chain.from_iterable(self.folder_img_list))
        img_indices = list(range(len(self.paths)))
        self.cls2indices = []
        self.index2cls   = []
        start_idx = 0

        for cls, img_list in enumerate(self.folder_img_list):
            end_idx = start_idx + len(img_list)
            self.cls2indices.append(img_indices[start_idx:end_idx])
            self.index2cls += [cls] * len(img_list)
            start_idx = end_idx

        print0("Found {} images in {}".format(len(self.paths), root))

    # Get list of folders with order as in map_file
    # Useful when we want to have the same splits (taking every n-th class)
    # Returns dictionary where key is folder name and value is label num as int
    # n02119789: 0
    # n02100735: 1
    # n02110185: 2
    # ...
    def get_folder_label_mapping(self):
        folders = []
        cls2label = []
        with open(self.map_file) as f:
            for line in f:
                tok = line.split()
                folders.append(tok[0])
                cls2label.append(tok[2])
        return folders, cls2label

    def save_example(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for cls, indices in enumerate(self.cls2indices):
            img_index = indices[0]
            img_dict = self.__getitem__(img_index)
            label = self.cls2label[cls]
            img_path = os.path.join(output_dir, f'{cls}-{label}.jpg')
            utils.save_image(img_dict['img'], img_path)

        print(f'{cls+1} example images saved to {output_dir}')

    '''
    def resize_dataset(self, new_folder):
        """
        create a resized dataset "imagenet128" and keep data folder structure.
        """
        for i in range(len(self.folder_img_list)):
            for j in range(len(self.folder_img_list[i])):
                old_path = self.folder_img_list[i][j]
                dirs = old_path.split('/')
                if self.split == 'test':
                    new_path = os.path.join(new_folder, dirs[1])
                else:
                    new_path = os.path.join(new_folder, dirs[1], dirs[2])
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                img = np.array(Image.open(old_path))
                img = aspect_preserving_resize(img, min_size=self.image_size)
                img = Image.fromarray(img)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(os.path.join(new_path, dirs[-1]))
    '''

class ClsByFolderDataset(BaseDataset):
    def __init__(self, root, image_size, exts = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG'], 
                 do_geo_aug=True, do_color_aug=True, training=True):
        super(ClsByFolderDataset, self).__init__(root, image_size, exts, do_geo_aug, do_color_aug, training)

        self.root = root
        # folder_img_list is a list of lists. Each sub-list is images in a folder (class).
        # 'train', 'val'
        self.folder_img_list = [ [ p for ext in exts \
            for p in sorted(glob.glob(os.path.join(self.root, folder_name) + f'/*.{ext}')) ] \
            for folder_name in os.listdir(self.root) ]

        self.paths = list(chain.from_iterable(self.folder_img_list))
        img_indices = list(range(len(self.paths)))
        self.cls2indices = []
        self.index2cls   = []
        start_idx = 0

        for cls, img_list in enumerate(self.folder_img_list):
            # Skip empty folders / non-folders
            if len(img_list) == 0:
                continue
            end_idx = start_idx + len(img_list)
            self.cls2indices.append(img_indices[start_idx:end_idx])
            self.index2cls += [cls] * len(img_list)
            start_idx = end_idx

        print0("Found {} images in {}".format(len(self.paths), root))

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

class DistributedDataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def sample_images(model, num_images, batch_size, dataset, img_save_path, nn_save_path):
    batches = num_to_groups(num_images, batch_size)

    old_rng_state = torch.random.get_rng_state()
    # Sampling uses independent noises and random seeds from training.
    torch.random.set_rng_state(model.sample_rng_state)
    
    # In all_images_list, each element is a batch of images.
    # In all_nn_list, if dataset is provided, each element is a batch of nearest neighbor images. 
    # Otherwise, all_nn_list is a list of None.
    all_images_nn_list = list(map(lambda n: model.sample(batch_size=n, dataset=dataset), batches))

    # Update sample_rng_state.
    model.sample_rng_state = torch.random.get_rng_state()
    # Restore the training random state.
    torch.random.set_rng_state(old_rng_state)

    all_images_list, all_nn_list = zip(*all_images_nn_list)
    all_images      = torch.cat(all_images_list, dim=0)
    utils.save_image(all_images,    img_save_path, nrow = 6)
    print0(f"Sampled {img_save_path}. ", end="")

    # If all_nn_list[0] is not None, then all elements in all_nn_list are not None.
    if exists(all_nn_list[0]):
        all_nn_images = torch.cat(all_nn_list, dim=0)
        utils.save_image(all_nn_images, nn_save_path, nrow = 6)
        print0(f"Nearest neighbors {nn_save_path}.")
    else:
        print0("")

# For CNN models, just forward to forward_features().
# For ViTs, patch the original timm code to keep the spatial dimensions of the extracted image features.
# use_head_feat: collapse geometric dimensions of the features.
def timm_extract_features(model, x, use_head_feat=False):
    # images have been normalized using normalize_to_neg_one_to_one(). 
    # Before applying the model's own normalization, we have to restore them to unnormalizd images.
    x = unnormalize_to_zero_to_one(x)
    # vit uses different mean/std from repvgg & resnet.
    img_mean = torch.tensor(model.default_cfg['mean'], device=x.device).view(1, 3, 1, 1)
    img_std  = torch.tensor(model.default_cfg['std'],  device=x.device).view(1, 3, 1, 1)
    # Imagenet-normalized x yields better features using an imagenet pretrained model.
    x = (x - img_mean) / img_std

    if type(model) != timm.models.vision_transformer.VisionTransformer:
        feat = model.forward_features(x)
        if use_head_feat:
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

    if use_head_feat:
        x = x[:, [0]].permute(0, 2, 1).reshape(x.shape[0], -1, 1, 1)
    else:
        # Remove the 'CLS' token from the output.
        x = x[:, 1:].permute(0, 2, 1).reshape(x.shape[0], -1, *out_size)
    return x

# extract_pre_feat(): extract features using a pretrained model.
# use_head_feat: use the features that feat_extractor uses to do the final classification. 
# It's not relevant to class embeddings used in as part of Unet features.
def extract_pre_feat(featnet_type, feat_extractor, img, ref_shape, has_grad=True, use_head_feat=False):
    if featnet_type == 'none':
        return None

    if featnet_type == 'mini':
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

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def unnorm_save_image(img, img_save_path, nrow, clip_denoised=True):
    if clip_denoised:
        img = dclamp(img, -1., 1.)

    img = unnormalize_to_zero_to_one(img)
    utils.save_image(img, img_save_path, nrow=nrow)

def l2norm(t):
    return F.normalize(t, dim = -1)

def fast_randn(*shape, device='cpu'):
    if type(shape[0]) == tuple:
        shape = shape[0]

    if str(device) == 'cpu':
        return torch.FloatTensor(*shape).normal_()
    else:
        return torch.cuda.FloatTensor(*shape).normal_()

def fast_randn_like(tens):
    return fast_randn(tens.shape, device=tens.device)


# Clamp with gradients to clamped elements.
# https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

# https://github.com/Aftaab99/pytorch-multiple-style-transfer/blob/master/transformer_net.py
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.inst_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, style_index):
        out = self.inst_norm(x)
        gamma, beta = self.embed(style_index).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out    

