# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract VAE latent features for paired MR-CT images.
This script pre-extracts latent space representations to save GPU memory during training.

The extracted features are saved as .npy files that can be loaded by train_mr_ct_latent.py.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os

from diffusers.models import AutoencoderKL


class MRCTDataset(Dataset):
    """
    Dataset for paired MR-CT images.
    """
    def __init__(
        self, 
        data_dir, 
        split='train',
        image_size=256,
        mr_mean=0.5,
        mr_std=0.5,
        ct_mean=0.5,
        ct_std=0.5
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
        mr_img = Image.open(mr_path).convert('L')
        
        # Load CT image (single channel/grayscale)
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
        
        return mr_tensor, ct_tensor, idx


def main(args):
    """
    Extract VAE latent features for MR-CT paired images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.features_path, exist_ok=True)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(args.features_path, split)
        os.makedirs(os.path.join(split_dir, 'mr_latent'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'ct_latent'), exist_ok=True)
    
    # Load VAE
    print(f"Loading VAE: stabilityai/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    # Process both train and test splits
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Check if split exists
        mr_dir = os.path.join(args.data_path, 'mr', split)
        ct_dir = os.path.join(args.data_path, 'ct', split)
        
        if not os.path.exists(mr_dir) or not os.path.exists(ct_dir):
            print(f"  Skipping {split} split (directory not found)")
            continue
        
        # Create dataset and dataloader
        dataset = MRCTDataset(
            data_dir=args.data_path,
            split=split,
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
        
        print(f"  Found {len(dataset)} paired images")
        
        split_dir = os.path.join(args.features_path, split)
        
        for mr_img, ct_img, indices in tqdm(loader, desc=f"  Extracting {split}"):
            mr_img = mr_img.to(device)
            ct_img = ct_img.to(device)
            
            with torch.no_grad():
                # Encode to latent space with scaling factor
                mr_latent = vae.encode(mr_img).latent_dist.sample().mul_(0.18215)
                ct_latent = vae.encode(ct_img).latent_dist.sample().mul_(0.18215)
            
            # Save as numpy files
            mr_latent = mr_latent.cpu().numpy()
            ct_latent = ct_latent.cpu().numpy()
            
            for i, idx in enumerate(indices):
                idx = idx.item()
                np.save(os.path.join(split_dir, 'mr_latent', f'{idx:06d}.npy'), mr_latent[i])
                np.save(os.path.join(split_dir, 'ct_latent', f'{idx:06d}.npy'), ct_latent[i])
    
    print(f"\nFeatures saved to {args.features_path}")
    print("Directory structure:")
    print(f"  {args.features_path}/")
    print(f"    train/")
    print(f"      mr_latent/  (MR latent features)")
    print(f"      ct_latent/  (CT latent features)")
    print(f"    test/")
    print(f"      mr_latent/")
    print(f"      ct_latent/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset directory containing 'mr' and 'ct' subdirectories")
    parser.add_argument("--features-path", type=str, default="features_mr_ct",
                        help="Output directory for extracted features")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    # Z-score normalization parameters
    parser.add_argument("--mr-mean", type=float, default=0.5)
    parser.add_argument("--mr-std", type=float, default=0.5)
    parser.add_argument("--ct-mean", type=float, default=0.5)
    parser.add_argument("--ct-std", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
