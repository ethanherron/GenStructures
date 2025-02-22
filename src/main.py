import argparse
import torch
from torch.utils.data import DataLoader
import wandb

from data import TopoDataset3D

from models.autoencoders import MultiHeadedVariationalAutoencoder
from trainers.multiheaded_vae import MultiHeadedVAETrainer
from models.dit import DiT
from trainers.latent_diffusion import LatentDiffusionTrainer
from trainers.utils import load_model


def train_vae(dataloader, device, num_steps=10000):
    wandb.init()
    
    model = MultiHeadedVariationalAutoencoder(
        in_channels=1,
        out_channels=1,
        dim=32,
        n_downsample=3,
        latent_channels=8,
        quantizer_beta=1.0,
        kl_beta=1e-4,
        inference=False
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(list(model.encoder_strain_energy.parameters()) + list(model.encoder_density.parameters()) + list(model.decoder.parameters()), lr=1e-4)
    
    trainer = MultiHeadedVAETrainer(model, dataloader, device, optimizer, num_steps)
    trainer.train(num_steps)


def train_ldm(dataloader, device, vae_weights_path, num_steps=10000):
    wandb.init()
    
    dit = DiT(
        input_size=8,
        patch_size=2,
        in_channels=8,
        out_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0
    )
    
    vae = MultiHeadedVariationalAutoencoder(
        in_channels=1,
        out_channels=1,
        dim=32,
        n_downsample=3,
        latent_channels=8,
        quantizer_beta=1.0,
        kl_beta=0.01,
        inference=True
    )
    
    vae = load_model(vae, vae_weights_path)
    
    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-4)
    
    trainer = LatentDiffusionTrainer(dit, vae, dataloader, device, optimizer, num_steps)
    trainer.train(num_steps)


def main():
    parser = argparse.ArgumentParser(description='Training script for VAE and LDM')
    parser.add_argument('mode', choices=['vae', 'ldm'], help='Choose training mode: "vae" or "ldm"')
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of steps for training')
    parser.add_argument('--vae_weights_path', type=str, default=None, help='Path to the VAE weights')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TopoDataset3D(args.data_path, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.mode == 'vae':
        train_vae(dataloader, device, num_steps=args.num_steps)
    elif args.mode == 'ldm':
        train_ldm(dataloader, device, args.vae_weights_path, num_steps=args.num_steps)


if __name__ == '__main__':
    main()
