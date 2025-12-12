# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for MR-to-CT synthesis using pre-extracted latent features.
This script loads pre-extracted VAE latent features to save GPU memory.

Use extract_features_mr_ct.py to pre-extract features before running this script.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models import DiT_models, DiT_MR_CT, load_pretrained_mr_ct
from diffusion import create_diffusion


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
#                             Latent Feature Dataset                            #
#################################################################################

class MRCTLatentDataset(Dataset):
    """
    Dataset for pre-extracted MR-CT latent features.
    
    Args:
        features_dir: Directory containing extracted features
                      (with 'mr_latent' and 'ct_latent' subdirectories)
        split: 'train' or 'test'
    """
    def __init__(self, features_dir, split='train'):
        self.features_dir = features_dir
        self.split = split
        
        # Get paths
        split_dir = os.path.join(features_dir, split)
        self.mr_latent_dir = os.path.join(split_dir, 'mr_latent')
        self.ct_latent_dir = os.path.join(split_dir, 'ct_latent')
        
        # Verify directories exist
        assert os.path.exists(self.mr_latent_dir), f"MR latent directory not found: {self.mr_latent_dir}"
        assert os.path.exists(self.ct_latent_dir), f"CT latent directory not found: {self.ct_latent_dir}"
        
        # Get sorted file lists
        self.mr_files = sorted([f for f in os.listdir(self.mr_latent_dir) if f.endswith('.npy')])
        self.ct_files = sorted([f for f in os.listdir(self.ct_latent_dir) if f.endswith('.npy')])
        
        # Verify same number of files
        assert len(self.mr_files) == len(self.ct_files), \
            f"Number of MR ({len(self.mr_files)}) and CT ({len(self.ct_files)}) latent files must match"
        
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        # Load pre-extracted latent features
        mr_latent = np.load(os.path.join(self.mr_latent_dir, self.mr_files[idx]))
        ct_latent = np.load(os.path.join(self.ct_latent_dir, self.ct_files[idx]))
        
        return torch.from_numpy(mr_latent), torch.from_numpy(ct_latent)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains DiT for MR-to-CT synthesis using pre-extracted latent features.
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
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-MR-CT-latent"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Training with pre-extracted latent features (no VAE in GPU memory)")

    # Create model:
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
    
    # Load pretrained weights if provided (for fine-tuning from scratch)
    if args.pretrained and not args.resume:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained weights from {args.pretrained}")
        model = load_pretrained_mr_ct(model, args.pretrained)
    
    model = model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    diffusion = create_diffusion(timestep_respacing="")
    
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Resume from checkpoint if specified
    train_steps = 0
    if args.resume:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        # Extract train_steps from checkpoint filename (format: 0000000.pt)
        ckpt_name = os.path.basename(args.resume)
        train_steps = int(ckpt_name.split('.')[0])
        if accelerator.is_main_process:
            logger.info(f"Resumed at step {train_steps}")

    # Setup data:
    dataset = MRCTLatentDataset(
        features_dir=args.features_path,
        split='train',
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
        logger.info(f"Dataset contains {len(dataset):,} paired latent features ({args.features_path})")

    # Prepare models for training:
    if not args.resume:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for mr_latent, ct_latent in loader:
            mr_latent = mr_latent.to(device)
            ct_latent = ct_latent.to(device)
            
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
                
                # Compute average loss across processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")
                avg_loss = avg_loss.item() / accelerator.num_processes

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
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path to pre-extracted features directory (from extract_features_mr_ct.py)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained DiT checkpoint (for fine-tuning from scratch)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from (restores model, ema, optimizer, and step count)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    args = parser.parse_args()
    main(args)
