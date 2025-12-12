# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for MR-to-CT synthesis using DiT.
This script fine-tunes a pretrained DiT model for paired MR-CT image translation.

The model takes concatenated (noisy CT latent + MR latent) as input (8 channels)
and outputs CT latent (4 channels + 4 for learned sigma).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from timm.models.vision_transformer import PatchEmbed

from models import DiT_models, DiT_MR_CT, load_pretrained_mr_ct
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                             MR-CT Dataset                                     #
#################################################################################

class MRCTDataset(Dataset):
    """
    Dataset for paired MR-CT images.
    
    Args:
        data_dir: Root directory containing 'mr' and 'ct' subdirectories
        split: 'train' or 'test'
        image_size: Target image size
        mr_mean: Mean for MR z-score normalization
        mr_std: Std for MR z-score normalization
        ct_mean: Mean for CT z-score normalization
        ct_std: Std for CT z-score normalization
    """
    def __init__(
        self, 
        data_dir, 
        split='train',
        image_size=256,
        mr_mean=0.0,
        mr_std=1.0,
        ct_mean=0.0,
        ct_std=1.0
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.mr_mean = mr_mean
        self.mr_std = mr_std
        self.ct_mean = ct_mean
        self.ct_std = ct_std
        
        # Get MR and CT image paths
        mr_dir = os.path.join(data_dir, 'mr', split)
        ct_dir = os.path.join(data_dir, 'ct', split)
        
        # Get sorted file lists
        self.mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Verify same number of files (paired data)
        assert len(self.mr_files) == len(self.ct_files), \
            f"Number of MR ({len(self.mr_files)}) and CT ({len(self.ct_files)}) images must match"
        
        self.mr_dir = mr_dir
        self.ct_dir = ct_dir
        
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        # Load MR image (single channel/grayscale)
        mr_path = os.path.join(self.mr_dir, self.mr_files[idx])
        mr_img = Image.open(mr_path).convert('L')  # Convert to grayscale
        
        # Load CT image (single channel/grayscale)
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_img = Image.open(ct_path).convert('L')  # Convert to grayscale
        
        # Resize to target size
        mr_img = mr_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        ct_img = ct_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        
        # Convert to numpy array and normalize to [0, 1]
        mr_arr = np.array(mr_img, dtype=np.float32) / 255.0
        ct_arr = np.array(ct_img, dtype=np.float32) / 255.0
        
        # Z-score normalization
        mr_arr = (mr_arr - self.mr_mean) / self.mr_std
        ct_arr = (ct_arr - self.ct_mean) / self.ct_std
        
        # Convert to tensor and add channel dimension
        # Shape: (1, H, W)
        mr_tensor = torch.from_numpy(mr_arr).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0)
        
        # Repeat to 3 channels for VAE (which expects RGB)
        # Shape: (3, H, W)
        mr_tensor = mr_tensor.repeat(3, 1, 1)
        ct_tensor = ct_tensor.repeat(3, 1, 1)
        
        return mr_tensor, ct_tensor


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains DiT for MR-to-CT synthesis.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-MR-CT"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    # Get model config from model name
    model_config = {
        'DiT-XL/2': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16, 'patch_size': 2},
        'DiT-XL/4': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16, 'patch_size': 4},
        'DiT-XL/8': {'hidden_size': 1152, 'depth': 28, 'num_heads': 16, 'patch_size': 8},
        'DiT-L/2': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 2},
        'DiT-L/4': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 4},
        'DiT-L/8': {'hidden_size': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 8},
        'DiT-B/2': {'hidden_size': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 2},
        'DiT-B/4': {'hidden_size': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 4},
        'DiT-B/8': {'hidden_size': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 8},
        'DiT-S/2': {'hidden_size': 384, 'depth': 12, 'num_heads': 6, 'patch_size': 2},
        'DiT-S/4': {'hidden_size': 384, 'depth': 12, 'num_heads': 6, 'patch_size': 4},
        'DiT-S/8': {'hidden_size': 384, 'depth': 12, 'num_heads': 6, 'patch_size': 8},
    }
    
    config = model_config[args.model]
    
    # Create DiT_MR_CT model
    model = DiT_MR_CT(
        input_size=latent_size,
        patch_size=config['patch_size'],
        in_channels=8,  # 4 noisy CT + 4 MR
        out_channels_base=4,  # CT latent has 4 channels
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
    )
    
    # Load pretrained weights if provided
    if args.pretrained:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained weights from {args.pretrained}")
        model = load_pretrained_mr_ct(model, args.pretrained)
    
    model = model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)  # Freeze VAE
    
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    dataset = MRCTDataset(
        data_dir=args.data_path,
        split='train',
        image_size=args.image_size,
        mr_mean=args.mr_mean,
        mr_std=args.mr_std,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} paired images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for mr_img, ct_img in loader:
            mr_img = mr_img.to(device)
            ct_img = ct_img.to(device)
            
            with torch.no_grad():
                # Encode MR and CT images to latent space
                # VAE expects input in range [-1, 1], we need to adjust for z-score normalized input
                # The z-score normalized values need to be clipped and scaled
                mr_latent = vae.encode(mr_img).latent_dist.sample().mul_(0.18215)
                ct_latent = vae.encode(ct_img).latent_dist.sample().mul_(0.18215)
            
            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (ct_latent.shape[0],), device=device)
            
            # Add noise to CT latent
            noise = torch.randn_like(ct_latent)
            noisy_ct_latent = diffusion.q_sample(ct_latent, t, noise=noise)
            
            # Concatenate noisy CT latent with MR latent as model input
            # Shape: (B, 8, H, W)
            x_input = torch.cat([noisy_ct_latent, mr_latent], dim=1)
            
            # Forward pass (DiT_MR_CT only takes x and t, no class labels)
            model_output = model(x_input, t)
            
            # Compute loss (predicting noise)
            # Model outputs 8 channels: 4 for epsilon, 4 for variance
            model_eps, model_var = torch.split(model_output, 4, dim=1)
            
            # MSE loss on epsilon prediction
            loss = torch.mean((model_eps - noise) ** 2)
            
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                avg_loss = torch.tensor(avg_loss, device=accelerator.device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, 
                        help="Path to dataset directory containing 'mr' and 'ct' subdirectories")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained DiT checkpoint")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    # Z-score normalization parameters
    parser.add_argument("--mr-mean", type=float, default=0.5,
                        help="Mean for MR z-score normalization")
    parser.add_argument("--mr-std", type=float, default=0.5,
                        help="Std for MR z-score normalization")
    parser.add_argument("--ct-mean", type=float, default=0.5,
                        help="Mean for CT z-score normalization")
    parser.add_argument("--ct-std", type=float, default=0.5,
                        help="Std for CT z-score normalization")
    args = parser.parse_args()
    main(args)
