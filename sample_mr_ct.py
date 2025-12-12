# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sampling script for MR-to-CT synthesis using DiT.
This script generates CT images from MR images using a trained DiT model.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

from models import DiT_MR_CT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             MR-CT Dataset (Test)                              #
#################################################################################

class MRCTTestDataset(Dataset):
    """
    Dataset for MR images during testing (with optional paired CT for comparison).
    
    Args:
        data_dir: Root directory containing 'mr' and 'ct' subdirectories
        split: 'train' or 'test'
        image_size: Target image size
        mr_mean: Mean for MR z-score normalization
        mr_std: Std for MR z-score normalization
        ct_mean: Mean for CT z-score normalization (for GT comparison)
        ct_std: Std for CT z-score normalization (for GT comparison)
    """
    def __init__(
        self, 
        data_dir, 
        split='test',
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
        
        # Get MR image paths
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
        mr_img = Image.open(mr_path).convert('L')
        
        # Load CT image (single channel/grayscale) for comparison
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        ct_img = Image.open(ct_path).convert('L')
        
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
        mr_tensor = torch.from_numpy(mr_arr).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0)
        
        # Repeat to 3 channels for VAE
        mr_tensor = mr_tensor.repeat(3, 1, 1)
        ct_tensor = ct_tensor.repeat(3, 1, 1)
        
        return mr_tensor, ct_tensor, self.mr_files[idx]


def denormalize_ct(ct_tensor, ct_mean, ct_std):
    """
    Denormalize CT tensor from z-score normalized values back to [0, 1] range.
    
    Args:
        ct_tensor: Tensor in z-score normalized space
        ct_mean: Mean used for z-score normalization
        ct_std: Std used for z-score normalization
        
    Returns:
        Tensor in [0, 1] range
    """
    # Reverse z-score normalization: x = z * std + mean
    ct_denorm = ct_tensor * ct_std + ct_mean
    # Clip to [0, 1] range
    ct_denorm = torch.clamp(ct_denorm, 0, 1)
    return ct_denorm


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
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
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    dataset = MRCTTestDataset(
        data_dir=args.data_path,
        split='test',
        image_size=args.image_size,
        mr_mean=args.mr_mean,
        mr_std=args.mr_std,
        ct_mean=args.ct_mean,
        ct_std=args.ct_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create output directory:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'comparison'), exist_ok=True)
    
    print(f"Generating CT images from {len(dataset)} MR images...")
    print(f"Saving results to {args.output_dir}")

    sample_idx = 0
    for mr_img, ct_gt, filenames in tqdm(loader, desc="Generating"):
        mr_img = mr_img.to(device)
        ct_gt = ct_gt.to(device)
        batch_size = mr_img.shape[0]
        
        # Encode MR image to latent space
        with torch.no_grad():
            mr_latent = vae.encode(mr_img).latent_dist.sample().mul_(0.18215)
        
        # Create initial noise for CT latent
        z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        
        # Custom forward function that concatenates MR latent with noisy CT latent
        def model_forward(x, t, **kwargs):
            # x is the noisy CT latent (4 channels)
            # Concatenate with MR latent to get 8-channel input
            x_input = torch.cat([x, mr_latent], dim=1)
            return model(x_input, t)
        
        # Sample CT latent using diffusion
        samples = diffusion.p_sample_loop(
            model_forward, 
            z.shape, 
            z, 
            clip_denoised=False, 
            model_kwargs={}, 
            progress=False, 
            device=device
        )
        
        # Decode CT latent to image
        samples = vae.decode(samples / 0.18215).sample
        
        # The VAE output is in [-1, 1] range, convert to [0, 1]
        samples = (samples + 1) / 2
        
        # Take the first channel (grayscale) and denormalize
        # Note: Since we used z-score normalized input to VAE, the output needs 
        # to be interpreted as z-score normalized values
        # For proper denormalization, we apply the inverse transformation
        # Here we assume the VAE output is already in a reasonable range
        samples_gray = samples.mean(dim=1, keepdim=True)  # Average RGB channels
        samples_denorm = denormalize_ct(samples_gray, args.ct_mean, args.ct_std)
        
        # Similarly process ground truth for comparison
        ct_gt_decoded = vae.decode(vae.encode(ct_gt).latent_dist.sample().mul_(0.18215) / 0.18215).sample
        ct_gt_vis = (ct_gt_decoded + 1) / 2
        ct_gt_gray = ct_gt_vis.mean(dim=1, keepdim=True)
        
        # Process MR for visualization
        mr_decoded = vae.decode(mr_latent / 0.18215).sample
        mr_vis = (mr_decoded + 1) / 2
        mr_gray = mr_vis.mean(dim=1, keepdim=True)
        
        # Save individual generated images
        for i in range(batch_size):
            filename = os.path.splitext(filenames[i])[0]
            
            # Save generated CT
            gen_path = os.path.join(args.output_dir, 'generated', f'{filename}_generated.png')
            save_image(samples_denorm[i], gen_path)
            
            # Save comparison (MR | Generated CT | Ground Truth CT)
            comparison = torch.cat([mr_gray[i], samples_denorm[i], ct_gt_gray[i]], dim=2)
            comp_path = os.path.join(args.output_dir, 'comparison', f'{filename}_comparison.png')
            save_image(comparison, comp_path)
            
            sample_idx += 1

    print(f"Done! Generated {sample_idx} CT images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset directory containing 'mr' and 'ct' subdirectories")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained MR-CT DiT checkpoint")
    parser.add_argument("--output-dir", type=str, default="samples_mr_ct",
                        help="Directory to save generated images")
    parser.add_argument("--model", type=str, choices=[
        'DiT-XL/2', 'DiT-XL/4', 'DiT-XL/8',
        'DiT-L/2', 'DiT-L/4', 'DiT-L/8',
        'DiT-B/2', 'DiT-B/4', 'DiT-B/8',
        'DiT-S/2', 'DiT-S/4', 'DiT-S/8',
    ], default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    # Z-score normalization parameters (should match training)
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
