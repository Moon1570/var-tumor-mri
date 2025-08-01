# VAR-Based Medical MRI Synthesis Pipeline
# Complete implementation for generating synthetic brain MRI images with tumors
# Based on Visual AutoRegressive (VAR) model architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import imageio
import cv2
import os
from glob import glob
import json
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ================================
# 1. Enhanced Data Preprocessing
# ================================

class BraTSPreprocessor:
    """Enhanced preprocessing for BraTS data with VAR-specific optimizations"""
    
    def __init__(self, input_dir: str, output_dir: str, target_size: int = 256):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        
        # Create output directories
        for subdir in ['images', 'masks', 'overlays', 'metadata']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Z-score normalization with clipping for stable training"""
        mean_val = np.mean(image[image > 0])  # Only non-zero voxels
        std_val = np.std(image[image > 0])
        
        if std_val == 0:
            return np.zeros_like(image)
        
        normalized = (image - mean_val) / std_val
        normalized = np.clip(normalized, -5, 5)  # Clip extreme values
        
        # Scale to [0, 255] for VAR training
        normalized = ((normalized - normalized.min()) / 
                     (normalized.max() - normalized.min()) * 255).astype(np.uint8)
        return normalized
    
    def resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio using padding"""
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target size
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        if len(image.shape) == 2:
            padded = np.pad(resized, ((pad_h, target_size - new_h - pad_h), 
                                    (pad_w, target_size - new_w - pad_w)), 
                          mode='constant', constant_values=0)
        else:
            padded = np.pad(resized, ((pad_h, target_size - new_h - pad_h), 
                                    (pad_w, target_size - new_w - pad_w), (0, 0)), 
                          mode='constant', constant_values=0)
        return padded
    
    def extract_tumor_slices(self) -> Dict[str, List[str]]:
        """Extract and preprocess FLAIR slices containing tumors"""
        h5_files = sorted(glob(os.path.join(self.input_dir, "*.h5")))
        metadata = {"files_processed": [], "tumor_slices": 0, "total_slices": 0}
        
        print(f"Processing {len(h5_files)} H5 files...")
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    image = f['image'][:]   # shape (240, 240, 4)
                    mask = f['mask'][:]     # shape (240, 240, 3)
                    
                    # Extract FLAIR (index 3) and check for tumors
                    flair = image[:, :, 3]
                    mask_combined = np.sum(mask, axis=-1)
                    
                    if np.any(mask_combined > 0):  # Has tumor
                        # Normalize FLAIR
                        flair_norm = self.normalize_intensity(flair)
                        
                        # Resize to target size
                        flair_resized = self.resize_with_padding(flair_norm, self.target_size)
                        mask_resized = self.resize_with_padding(mask_combined, self.target_size)
                        
                        # Save image
                        filename = os.path.splitext(os.path.basename(h5_file))[0]
                        img_path = os.path.join(self.output_dir, "images", f"{filename}.png")
                        mask_path = os.path.join(self.output_dir, "masks", f"{filename}.png")
                        
                        imageio.imwrite(img_path, flair_resized)
                        imageio.imwrite(mask_path, (mask_resized > 0).astype(np.uint8) * 255)
                        
                        # Create overlay for visualization
                        overlay = cv2.cvtColor(flair_resized, cv2.COLOR_GRAY2BGR)
                        overlay[mask_resized > 0] = [255, 0, 0]  # Red tumor regions
                        overlay_path = os.path.join(self.output_dir, "overlays", f"{filename}.png")
                        cv2.imwrite(overlay_path, overlay)
                        
                        metadata["tumor_slices"] += 1
                        metadata["files_processed"].append(filename)
                    
                    metadata["total_slices"] += 1
                    
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")
                continue
        
        # Save metadata
        with open(os.path.join(self.output_dir, "metadata", "preprocessing_stats.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Extracted {metadata['tumor_slices']} tumor slices from {metadata['total_slices']} total slices")
        return metadata

# ================================
# 2. VAR Model Architecture
# ================================

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VAR"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, (perplexity, encodings, encoding_indices)

class VQVAEEncoder(nn.Module):
    """Multi-scale encoder for VAR"""
    
    def __init__(self, in_channels: int = 1, hidden_dims: List[int] = [64, 128, 256, 512], 
                 embedding_dim: int = 256):
        super().__init__()
        
        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.pre_quant_conv = nn.Conv2d(hidden_dims[-1], embedding_dim, 1)
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.pre_quant_conv(encoded)

class VQVAEDecoder(nn.Module):
    """Multi-scale decoder for VAR"""
    
    def __init__(self, embedding_dim: int = 256, hidden_dims: List[int] = [512, 256, 128, 64], 
                 out_channels: int = 1):
        super().__init__()
        
        self.post_quant_conv = nn.Conv2d(embedding_dim, hidden_dims[0], 1)
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        modules.append(nn.ConvTranspose2d(hidden_dims[-1], out_channels, 
                                        kernel_size=4, stride=2, padding=1))
        modules.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.post_quant_conv(x)
        return self.decoder(x)

class VQVAE(nn.Module):
    """Complete VQ-VAE for multi-scale image tokenization"""
    
    def __init__(self, in_channels: int = 1, embedding_dim: int = 256, 
                 num_embeddings: int = 8192, commitment_cost: float = 0.25):
        super().__init__()
        
        self.encoder = VQVAEEncoder(in_channels, embedding_dim=embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(embedding_dim, out_channels=in_channels)
    
    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, vq_info = self.vq_layer(encoded)
        decoded = self.decoder(quantized)
        
        return decoded, vq_loss, vq_info

class MultiHeadAttention(nn.Module):
    """Multi-head attention for VAR transformer"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class VARTransformerBlock(nn.Module):
    """Transformer block for VAR autoregressive model"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class VARModel(nn.Module):
    """Visual AutoRegressive model for next-scale prediction"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, d_ff: int = 3072, max_seq_len: int = 1024, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            VARTransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens, targets=None):
        batch_size, seq_len = tokens.shape
        
        # Embeddings
        token_emb = self.token_embedding(tokens)
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return logits

# ================================
# 3. Dataset and Training
# ================================

class MRIDataset(Dataset):
    """Dataset for MRI images"""
    
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = imageio.imread(img_path)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        return torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

class VARTrainer:
    """Training pipeline for VAR model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.vqvae = VQVAE(
            in_channels=1,
            embedding_dim=config['embedding_dim'],
            num_embeddings=config['num_embeddings']
        ).to(self.device)
        
        self.var_model = VARModel(
            vocab_size=config['num_embeddings'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        ).to(self.device)
        
        # Optimizers
        self.vqvae_optimizer = torch.optim.AdamW(
            self.vqvae.parameters(), 
            lr=config['vqvae_lr'], 
            weight_decay=config['weight_decay']
        )
        
        self.var_optimizer = torch.optim.AdamW(
            self.var_model.parameters(), 
            lr=config['var_lr'], 
            weight_decay=config['weight_decay']
        )
        
        # Data
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
        
        dataset = MRIDataset(config['data_path'], transform=transform)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4
        )
        
        print(f"Dataset loaded: {len(dataset)} images")
        print(f"Training on device: {self.device}")
    
    def train_vqvae(self, epochs: int):
        """Train VQ-VAE for tokenization"""
        print("Training VQ-VAE...")
        
        self.vqvae.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_vq_loss = 0
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                
                self.vqvae_optimizer.zero_grad()
                
                reconstructed, vq_loss, _ = self.vqvae(batch)
                recon_loss = F.mse_loss(reconstructed, batch)
                
                loss = recon_loss + vq_loss
                loss.backward()
                self.vqvae_optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, "
                          f"VQ={vq_loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(self.vqvae.state_dict(), 
                          f"checkpoints/vqvae_epoch_{epoch}.pth")
    
    def extract_tokens(self):
        """Extract tokens from trained VQ-VAE"""
        print("Extracting tokens...")
        
        self.vqvae.eval()
        all_tokens = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                batch = batch.to(self.device)
                encoded = self.vqvae.encoder(batch)
                _, _, (_, _, tokens) = self.vqvae.vq_layer(encoded)
                all_tokens.append(tokens.cpu())
        
        return torch.cat(all_tokens, dim=0)
    
    def train_var(self, epochs: int):
        """Train VAR autoregressive model"""
        print("Training VAR...")
        
        # First extract tokens
        tokens = self.extract_tokens()
        
        # Create token dataset
        token_dataset = torch.utils.data.TensorDataset(tokens)
        token_dataloader = DataLoader(
            token_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        self.var_model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (batch_tokens,) in enumerate(token_dataloader):
                batch_tokens = batch_tokens.to(self.device).squeeze(-1)
                
                # Prepare inputs and targets for autoregressive training
                inputs = batch_tokens[:, :-1]
                targets = batch_tokens[:, 1:]
                
                self.var_optimizer.zero_grad()
                
                logits, loss = self.var_model(inputs, targets)
                loss.backward()
                self.var_optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}")
            
            avg_loss = total_loss / len(token_dataloader)
            print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(self.var_model.state_dict(), 
                          f"checkpoints/var_epoch_{epoch}.pth")

# ================================
# 4. Generation and Evaluation
# ================================

class MRIGenerator:
    """Generate synthetic MRI images using trained VAR"""
    
    def __init__(self, vqvae_path: str, var_path: str, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load models
        self.vqvae = VQVAE(
            in_channels=1,
            embedding_dim=config['embedding_dim'],
            num_embeddings=config['num_embeddings']
        ).to(self.device)
        self.vqvae.load_state_dict(torch.load(vqvae_path, map_location=self.device))
        
        self.var_model = VARModel(
            vocab_size=config['num_embeddings'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        ).to(self.device)
        self.var_model.load_state_dict(torch.load(var_path, map_location=self.device))
        
        self.vqvae.eval()
        self.var_model.eval()
    
    def generate(self, num_samples: int, temperature: float = 1.0, 
                top_k: int = 50) -> List[np.ndarray]:
        """Generate synthetic MRI images"""
        print(f"Generating {num_samples} synthetic MRI images...")
        
        generated_images = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Start with a seed token
                tokens = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                
                # Generate tokens autoregressively
                for _ in range(self.config['max_tokens'] - 1):
                    logits = self.var_model(tokens)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[top_k_indices] = top_k_logits
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).unsqueeze(0)
                    tokens = torch.cat([tokens, next_token], dim=1)
                
                # Decode tokens to image
                # Note: This requires reshaping tokens to match encoder output shape
                # Implementation depends on specific tokenization scheme
                token_map = tokens[0, 1:].view(1, 16, 16)  # Adjust dimensions as needed
                
                # Convert tokens back to embeddings and decode
                embeddings = self.vqvae.vq_layer.embeddings(token_map.long())
                embeddings = embeddings.permute(0, 3, 1, 2)  # BHWC -> BCHW
                
                generated_image = self.vqvae.decoder(embeddings)
                generated_image = generated_image.squeeze().cpu().numpy()
                generated_image = (generated_image * 255).astype(np.uint8)
                
                generated_images.append(generated_image)
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_samples} images")
        
        return generated_images
    
    def save_generated_images(self, images: List[np.ndarray], output_dir: str):
        """Save generated images"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            filename = f"synthetic_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            imageio.imwrite(filepath, img)
        
        print(f"Saved {len(images)} synthetic images to {output_dir}")

# ================================
# 5. Main Training Pipeline
# ================================

def main():
    parser = argparse.ArgumentParser(description="VAR-based MRI synthesis")
    parser.add_argument("--mode", choices=["preprocess", "train", "generate"], 
                       required=True, help="Mode to run")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to BraTS data or processed images")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Output directory")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        "embedding_dim": 256,
        "num_embeddings": 8192,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "batch_size": 8,
        "vqvae_lr": 1e-4,
        "var_lr": 1e-4,
        "weight_decay": 0.01,
        "max_tokens": 256,
        "target_size": 256
    }
    
    # Load config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.mode == "preprocess":
        print("Preprocessing BraTS data...")
        preprocessor = BraTSPreprocessor(args.data_path, args.output_path, 
                                       config['target_size'])
        metadata = preprocessor.extract_tumor_slices()
        print(f"Preprocessing complete. {metadata['tumor_slices']} tumor slices extracted.")
    
    elif args.mode == "train":
        print("Training VAR model...")
        config['data_path'] = os.path.join(args.data_path, "images")
        
        trainer = VARTrainer(config)
        
        # Train VQ-VAE first
        print("Phase 1: Training VQ-VAE...")
        trainer.train_vqvae(epochs=50)
        
        # Train VAR model
        print("Phase 2: Training VAR...")
        trainer.train_var(epochs=100)
        
        print("Training complete!")
    
    elif args.mode == "generate":
        print("Generating synthetic MRI images...")
        
        vqvae_path = "checkpoints/vqvae_epoch_40.pth"  # Adjust path as needed
        var_path = "checkpoints/var_epoch_90.pth"     # Adjust path as needed
        
        if not os.path.exists(vqvae_path) or not os.path.exists(var_path):
            print("Error: Model checkpoints not found. Please train models first.")
            return
        
        generator = MRIGenerator(vqvae_path, var_path, config)
        synthetic_images = generator.generate(num_samples=100, temperature=0.8, top_k=50)
        generator.save_generated_images(synthetic_images, args.output_path)
        
        print("Generation complete!")

# ================================
# 6. Evaluation and Quality Assessment
# ================================

class MRIQualityAssessment:
    """Quality assessment tools for synthetic MRI images"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calculate_fid(self, real_images: List[np.ndarray], 
                     synthetic_images: List[np.ndarray]) -> float:
        """Calculate Fréchet Inception Distance (FID) between real and synthetic images"""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            
            fid = FrechetInceptionDistance(feature=2048).to(self.device)
            
            # Convert images to tensors
            real_tensors = []
            synthetic_tensors = []
            
            for img in real_images:
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=2)  # Convert to RGB
                tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
                real_tensors.append(tensor)
            
            for img in synthetic_images:
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=2)  # Convert to RGB
                tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
                synthetic_tensors.append(tensor)
            
            real_batch = torch.cat(real_tensors).to(self.device)
            synthetic_batch = torch.cat(synthetic_tensors).to(self.device)
            
            fid.update(real_batch, real=True)
            fid.update(synthetic_batch, real=False)
            
            return fid.compute().item()
        
        except ImportError:
            print("Warning: torchmetrics not available. FID calculation skipped.")
            return -1.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM) between two images"""
        from skimage.metrics import structural_similarity as ssim
        
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        return ssim(img1, img2, data_range=255)
    
    def assess_image_quality(self, images: List[np.ndarray]) -> Dict[str, float]:
        """Comprehensive quality assessment of synthetic images"""
        metrics = {
            "mean_intensity": [],
            "std_intensity": [],
            "contrast": [],
            "sharpness": []
        }
        
        for img in images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Basic statistics
            metrics["mean_intensity"].append(np.mean(img))
            metrics["std_intensity"].append(np.std(img))
            
            # Contrast (standard deviation of pixel intensities)
            metrics["contrast"].append(np.std(img))
            
            # Sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            metrics["sharpness"].append(np.var(laplacian))
        
        # Calculate averages
        quality_report = {}
        for key, values in metrics.items():
            quality_report[f"avg_{key}"] = np.mean(values)
            quality_report[f"std_{key}"] = np.std(values)
        
        return quality_report
    
    def generate_quality_report(self, real_images: List[np.ndarray], 
                              synthetic_images: List[np.ndarray], 
                              output_path: str):
        """Generate comprehensive quality assessment report"""
        print("Generating quality assessment report...")
        
        # Calculate metrics
        fid_score = self.calculate_fid(real_images, synthetic_images)
        real_quality = self.assess_image_quality(real_images)
        synthetic_quality = self.assess_image_quality(synthetic_images)
        
        # Calculate average SSIM between random pairs
        ssim_scores = []
        for i in range(min(50, len(synthetic_images))):
            for j in range(min(50, len(real_images))):
                ssim_score = self.calculate_ssim(synthetic_images[i], real_images[j])
                ssim_scores.append(ssim_score)
        
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
        
        # Generate report
        report = {
            "fid_score": fid_score,
            "average_ssim": avg_ssim,
            "real_image_quality": real_quality,
            "synthetic_image_quality": synthetic_quality,
            "num_real_images": len(real_images),
            "num_synthetic_images": len(synthetic_images)
        }
        
        # Save report
        with open(os.path.join(output_path, "quality_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Quality Assessment Summary:")
        print(f"FID Score: {fid_score:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Real Images - Avg Contrast: {real_quality['avg_contrast']:.2f}")
        print(f"Synthetic Images - Avg Contrast: {synthetic_quality['avg_contrast']:.2f}")
        
        return report

# ================================
# 7. Downstream Task Training
# ================================

class TumorSegmentationModel(nn.Module):
    """Simple U-Net for tumor segmentation using synthetic data"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d4 = self.upconv4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        return torch.sigmoid(self.final(d2))

class DownstreamTrainer:
    """Train tumor segmentation model on synthetic data"""
    
    def __init__(self, synthetic_data_path: str, real_test_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TumorSegmentationModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()
        
        # Load synthetic training data
        self.train_dataset = self._load_synthetic_dataset(synthetic_data_path)
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        
        # Load real test data
        self.test_dataset = self._load_real_dataset(real_test_path)
        self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=False)
        
        print(f"Training on {len(self.train_dataset)} synthetic samples")
        print(f"Testing on {len(self.test_dataset)} real samples")
    
    def _load_synthetic_dataset(self, data_path: str):
        """Load synthetic images with manual annotations"""
        # This would load synthetic images and their corresponding masks
        # For demonstration, we create a simple dataset structure
        images = []
        masks = []
        
        image_files = sorted(glob(os.path.join(data_path, "images", "*.png")))
        mask_files = sorted(glob(os.path.join(data_path, "masks", "*.png")))
        
        for img_file, mask_file in zip(image_files, mask_files):
            img = imageio.imread(img_file).astype(np.float32) / 255.0
            mask = imageio.imread(mask_file).astype(np.float32) / 255.0
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            images.append(torch.FloatTensor(img).unsqueeze(0))
            masks.append(torch.FloatTensor(mask).unsqueeze(0))
        
        return list(zip(images, masks))
    
    def _load_real_dataset(self, data_path: str):
        """Load real test images with ground truth masks"""
        # Similar to synthetic dataset loading but for real data
        images = []
        masks = []
        
        image_files = sorted(glob(os.path.join(data_path, "images", "*.png")))
        mask_files = sorted(glob(os.path.join(data_path, "masks", "*.png")))
        
        for img_file, mask_file in zip(image_files, mask_files):
            img = imageio.imread(img_file).astype(np.float32) / 255.0
            mask = imageio.imread(mask_file).astype(np.float32) / 255.0
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            images.append(torch.FloatTensor(img).unsqueeze(0))
            masks.append(torch.FloatTensor(mask).unsqueeze(0))
        
        return list(zip(images, masks))
    
    def train(self, epochs: int):
        """Train segmentation model"""
        print("Training tumor segmentation model...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images = torch.stack(images).to(self.device)
                masks = torch.stack(masks).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}")
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
            
            # Evaluate on test set
            if epoch % 5 == 0:
                self.evaluate()
    
    def evaluate(self):
        """Evaluate on real test data"""
        self.model.eval()
        total_dice = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, masks in self.test_loader:
                images = torch.stack(images).to(self.device)
                masks = torch.stack(masks).to(self.device)
                
                outputs = self.model(images)
                predictions = (outputs > 0.5).float()
                
                # Calculate Dice coefficient
                for pred, mask in zip(predictions, masks):
                    dice = self._dice_coefficient(pred, mask)
                    total_dice += dice
                    total_samples += 1
        
        avg_dice = total_dice / total_samples if total_samples > 0 else 0
        print(f"Test Dice Score: {avg_dice:.4f}")
        
        self.model.train()
        return avg_dice
    
    def _dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient"""
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = torch.sum(pred * target)
        total = torch.sum(pred) + torch.sum(target)
        
        if total == 0:
            return 1.0
        
        dice = (2.0 * intersection) / total
        return dice.item()

# ================================
# 8. Complete Usage Example
# ================================

def run_complete_pipeline():
    """Example of running the complete VAR-MRI pipeline"""
    
    # Configuration
    config = {
        "embedding_dim": 256,
        "num_embeddings": 4096,  # Reduced for faster training
        "d_model": 512,          # Reduced for faster training
        "num_layers": 8,         # Reduced for faster training
        "num_heads": 8,
        "batch_size": 4,         # Reduced for memory constraints
        "vqvae_lr": 1e-4,
        "var_lr": 1e-4,
        "weight_decay": 0.01,
        "max_tokens": 256,
        "target_size": 256
    }
    
    # Save configuration
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=== VAR-MRI Complete Pipeline ===")
    print("This pipeline will:")
    print("1. Preprocess BraTS data")
    print("2. Train VQ-VAE for tokenization")
    print("3. Train VAR for generation")
    print("4. Generate synthetic MRI images")
    print("5. Evaluate image quality")
    print("6. Train downstream tumor segmentation model")
    print()
    
    # Example usage - adjust paths as needed
    raw_data_path = "brats2020-kaggle/BraTS2020__training_data/content/data"
    processed_data_path = "processed_data"
    synthetic_output_path = "synthetic_images"
    
    print("Run the pipeline with:")
    print(f"python var_mri_pipeline.py --mode preprocess --data_path {raw_data_path} --output_path {processed_data_path}")
    print(f"python var_mri_pipeline.py --mode train --data_path {processed_data_path} --output_path checkpoints")
    print(f"python var_mri_pipeline.py --mode generate --data_path checkpoints --output_path {synthetic_output_path}")

if __name__ == "__main__":
    main()