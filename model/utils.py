import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import timm
from .laplacian import LapLoss

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

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

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
def timm_extract_features(model, x):
    if type(model) != timm.models.vision_transformer.VisionTransformer:
        return model.forward_features(x)

    # img_size and patch_size are tuples.
    img_size = model.patch_embed.img_size
    patch_size = model.patch_embed.patch_size
    out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

    # timm vit only accepts fixed images. So x has to be resized to [224, 224].
    if x.shape[2:] != img_size:
        x = torch.nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=False)

    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    x = model.blocks(x)
    x = model.norm(x)
    # Remove the 'CLS' token from the output.
    x = x[:, 1:].permute(0, 2, 1).reshape(x.shape[0], -1, *out_size)
    return x


# Dual teaching helps slightly.
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
