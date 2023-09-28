import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# Library
import os
from os.path import join as opj
from glob import glob
import sys

import random
import numpy as np

import cv2
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

import segmentation_models_pytorch as smp
#from torch_poly_lr_decay import PolynomialLRDecay

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#### Dataset ####
class LG_dataset(Dataset):
    def __init__(self, args, fold:int=0, mode="train", shuffle=False):
        if args.r_step is None:
            self.input_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}", "train/input", "*.png")))
        else:  # patch overlap
            self.input_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}_{args.r_step}x{args.c_step}", "train/input", "*.png")))

        if fold == 0:  # no validation
            self.val_data = 0
            print("no validation")
        else:  # CV
            self.n_val = len(self.input_paths) // args.n_folds
            if mode == "train":
                self.input_transform = args.train_input_transform
                self.label_transform = args.train_label_transform
                self.input_paths = self.input_paths[:(fold-1)*self.n_val] + self.input_paths[fold*self.n_val:]
            elif mode == "valid":
                self.input_transform = args.valid_input_transform
                self.label_transform = args.valid_label_transform
                self.input_paths = self.input_paths[(fold-1)*self.n_val:fold*self.n_val]
        if shuffle:
            random.shuffle(self.input_paths)
        self.label_paths = [x.replace("input", "label") for x in self.input_paths]
        print(f"[{mode}] inputs: {len(self.input_paths)} labels: {len(self.label_paths)}")
        
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        label_path = self.label_paths[idx]
        
        input_img = Image.open(input_path).convert("RGB")
        label_img = Image.open(label_path).convert("RGB")
        input_img = self.input_transform(input_img)
        label_img = self.label_transform(label_img)
        # return input_img, label_img
        return input_img, label_img

class LG_dataset_team(Dataset):
    def __init__(self, args, mode="train", shuffle=True):
        self.mode = mode  # train 일때만 augmentation
        
        if args.r_step is None:
            self.input_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}", mode, "input/*.png")))
        else:
            self.input_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}_{args.r_step}x{args.c_step}", mode, "input/*.png")))
                                               
        if shuffle:
            random.shuffle(self.input_paths)
        self.label_paths = [x.replace("input", "label") for x in self.input_paths]
        print(f"[{mode}] input paths: {len(self.input_paths)}, label paths: {len(self.label_paths)}")
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        label_path = self.label_paths[idx]
        
        input_img = np.asarray(Image.open(input_path).convert("RGB")) / 255.0
        label_img = np.asarray(Image.open(label_path).convert("RGB")) / 255.0
        
        if "resize" in args.train_transform:
            input_img = cv2.resize(input_img, (args.input_c_size, args.input_r_size))
                  
        if random.random() < 0.5 and self.mode == "train" and "vert_flip" in args.train_transform:
            input_img = input_img[:, ::-1, :]
            label_img = label_img[:, ::-1, :]
        
        if random.random() < 0.5 and self.mode == "train" and "hori_flip" in args.train_transform:
            input_img = input_img[::-1, :, :]
            label_img = label_img[::-1, :, :]
        input_img = torch.from_numpy(input_img.copy()).float()
        label_img = torch.from_numpy(label_img.copy()).float()
        input_img = input_img.permute(2,0,1)
        label_img = label_img.permute(2,0,1)
        
        return input_img, label_img
    
class LG_dataset_test(Dataset):
    def __init__(self, args):
        self.args = args
        if args.r_step is None:
            self.img_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}", "test/blended/*.png")))
        else:
            self.img_paths = sorted(glob(opj(args.data_root_dir, f"{args.r_size}x{args.c_size}_{args.r_step}x{args.c_step}", "test/blended/*.png")))
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        dir_, file_name = os.path.split(img_path)
        img = cv2.imread(img_path)[:,:,::-1] / 255.0
        img = cv2.resize(img, (self.args.input_c_size, self.args.input_r_size))
        img = torch.from_numpy(img.copy()).float()
        img = img.permute(2,0,1)
        return img, file_name
        


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


#### utils ####
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None
        
    def open(self, file_path, mode=None):
        if mode is None: mode='w'
        self.file = open(file_path, mode)
        
    def write(self, msg, is_terminal=1, is_file=1):
        if '\r' in msg: is_file=0
            
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
            
        if is_file == 1:
            self.file.write(msg)
            self.file.flush()
            
    def flush(self):
        pass
    
def fix_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True # this can slow down speed
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def calc_psnr(img1, img2):
    if torch.equal(img1, img2):
        return 100
    else:
        mse = torch.mean((img1-img2)**2)
    return 10 * torch.log10(1/mse)

# ----------
#  Training
# ----------

class args:
    #### config ####
    cuda_visible_devices = "1"
    seed=42
    
    #### dataset ####
    data_root_dir = "../data/LG/patch"
    r_size=612
    c_size=816
#     input_r_size = 608
#     input_c_size = 800
    input_r_size = 640
    input_c_size = 832
    r_step = 306
    c_step = 408
    batch_size=16
    num_workers=8
    
    train_transform = ["resize", "vert_flip", "hori_flip"]

    #### train ####
    n_epochs = 300
    G_lr = 2e-4
    G_betas = (0.5, 0.999)
    G_alpha_pixel = 100
    D_lr = 1e-4
    D_betas = (0.5, 0.999)
    
    lr_decay_ratio = 0.1
    betas=(0.5,0.999)
    
    scheduler_patience=10
    
    #### log & save ####
    save_dir = f"./saved_models/GAN_[r size-{r_size}]_[c size-{c_size}]_[input r size-{input_r_size}]_[input c size-{input_c_size}]_[G lr-{G_lr}]_[D lr-{D_lr}]"
    logger_path = opj(save_dir, "log.txt")
    
    G_load_path = "./saved_models/GAN_[r size-612]_[c size-816]_[input r size-640]_[input c size-832]_[G lr-0.0002]_[D lr-0.0001]/saved_models/G_167_29.627469539642334.pth"
    D_load_path = "./saved_models/GAN_[r size-612]_[c size-816]_[input r size-640]_[input c size-832]_[G lr-0.0002]_[D lr-0.0001]/saved_models/D_167_29.627469539642334.pth"
    
    #### test ####
    test_dir = "./submission_GAN"
    
train_img_save_dir = opj(args.save_dir, "train_saved_images")
valid_img_save_dir = opj(args.save_dir, "valid_saved_images")
model_save_dir = opj(args.save_dir, "saved_models")
os.makedirs(train_img_save_dir, exist_ok=True)
os.makedirs(valid_img_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

prev_time = time.time()
fix_seed(args.seed)
logger = Logger()
os.makedirs(args.save_dir, exist_ok=True)
logger.open(args.logger_path)

real_label = Variable(torch.ones(args.batch_size, 1, args.input_r_size // 16, args.input_c_size // 16), requires_grad=False).cuda()
gene_label = Variable(torch.zeros(args.batch_size, 1, args.input_r_size // 16, args.input_c_size // 16), requires_grad=False).cuda()

train_dataset = LG_dataset_team(args, mode="train", shuffle=True)
valid_dataset = LG_dataset_team(args, mode="validation", shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers//2, pin_memory=True, drop_last=True)
T_resize = T.Resize((args.input_r_size, args.input_c_size))
T_resize_back = T.Resize((args.r_size, args.c_size))

G = nn.DataParallel(GeneratorUNet().cuda())
D = nn.DataParallel(Discriminator().cuda())

criterion_MSE = nn.MSELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr, betas=args.G_betas)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.D_lr, betas=args.D_betas)



start_epoch = 0
if os.path.isfile(args.G_load_path):
    G_state_dict = torch.load(args.G_load_path)
    G.load_state_dict(G_state_dict["G_state_dict"])
    optimizer_G.load_state_dict(G_state_dict["optimizer_G"])
    start_epoch = G_state_dict["epoch"]
    logger.write("G load success!!!!\n")
if os.path.isfile(args.D_load_path):
    D_state_dict = torch.load(args.D_load_path)
    D.load_state_dict(D_state_dict["D_state_dict"])
    optimizer_D.load_state_dict(D_state_dict["optimizer_D"])
    logger.write("D load success!!!!\n")

# amp
scaler = GradScaler()

best_psnr=0
for epoch in range(start_epoch, args.n_epochs):
    G.train()
    D.train()
    G_loss = 0
    D_loss = 0
    valid_psnr = 0
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for iter_, (input_img, label_img) in train_loop:
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        input_img = input_img.cuda()
        resized_label_img = T_resize(label_img).cuda()
        label_img = label_img.cuda()
        
        with autocast():
            gene_img = G(input_img)
            # train G
            gene_preds = D(gene_img, resized_label_img)
            G_pixel_loss_ = criterion_MSE(resized_label_img, gene_img)
            G_adv_loss_ = criterion_L1(gene_preds, real_label)
            G_loss_ = G_adv_loss_ + args.G_alpha_pixel * G_pixel_loss_
        
            # train D
            real_preds = D(input_img, resized_label_img)
            gene_preds = D(gene_img.detach(), resized_label_img)
            D_real_loss_ = criterion_L1(real_preds, real_label)
            D_gene_loss_ = criterion_L1(gene_preds, gene_label)
            D_loss_ = (D_real_loss_ + D_gene_loss_) / 2
            
        scaler.scale(G_loss_).backward()
        scaler.scale(D_loss_).backward()
        scaler.step(optimizer_G)
        scaler.step(optimizer_D)
        scaler.update()
        
        

        
        resized_gene_img = T_resize_back(gene_img)  # 실제로 나가는 아웃풋 이미지.
        G_loss += G_loss_.item()
        D_loss += D_loss_.item()
        
    for i, img in enumerate(resized_gene_img.detach().cpu().numpy()):
        to_path = opj(train_img_save_dir, f"{epoch}_{i}.png")
        cv2.imwrite(to_path, (img.transpose(1,2,0)*255).astype("uint8"))
    # validation
    G.eval()
    D.eval()
    with torch.no_grad():
        valid_loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
        for iter_, (input_img, label_img) in valid_loop:
            input_img = input_img.cuda()
            resized_label_img = T_resize(label_img).cuda()
            label_img = label_img.cuda()
            
            with autocast():
                gene_img = G(input_img)
            resized_gene_img = T_resize_back(gene_img)
            psnr_ = calc_psnr(resized_gene_img, label_img)
            valid_psnr += psnr_.item()
                
    for i, img in enumerate(resized_gene_img.detach().cpu().numpy()):
        to_path = opj(valid_img_save_dir, f"{epoch}_{i}.png")
        cv2.imwrite(to_path, (img.transpose(1,2,0)*255).astype("uint8"))
        
    cur_psnr = valid_psnr / len(valid_loader)
    if best_psnr < cur_psnr:
        G_state_dict = {"G_state_dict": G.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "epoch": epoch}
        D_state_dict = {"D_state_dict": D.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                        "epoch": epoch}
        G_to_path = opj(model_save_dir, f"G_{epoch}_{best_psnr}.pth")
        D_to_path = opj(model_save_dir, f"D_{epoch}_{best_psnr}.pth")
        update_msg = f"best psnr update: {best_psnr} -> {cur_psnr}, model save....\n"   
        best_psnr = cur_psnr
        torch.save(G_state_dict, G_to_path)
        torch.save(D_state_dict, D_to_path)
        logger.write(update_msg)
    log_msg = f"[Epoch-[{epoch}/{args.n_epochs}]]_[train G loss-{G_loss}]_[train D loss-{D_loss}]_[valid_psnr-{valid_psnr/len(valid_loader)}]_[]_[]\n"
    logger.write(log_msg)