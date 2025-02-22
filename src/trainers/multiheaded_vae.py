import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from models.autoencoders import MultiHeadedVariationalAutoencoder
from .utils import save_models

class MultiHeadedVAETrainer:
    def __init__(self, 
                 model: MultiHeadedVariationalAutoencoder, 
                 train_loader: DataLoader, 
                 device: torch.device, 
                 optimizer: torch.optim.Optimizer, 
                 num_steps: int
                 ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.scaler = GradScaler()
        
        self.model.to(self.device)
    
    def train(self, num_steps: int):
        self.model.train()
        loader = iter(self.train_loader)
        
        for step in tqdm(range(num_steps)):
            try:
                strain_energy, density = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                strain_energy, density = next(loader)
                
            strain_energy = strain_energy.to(self.device)
            density = density.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                loss, reconstruction_loss, quantization_loss, continuous_loss = self.model(strain_energy, density)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            wandb.log({
                'vae/loss': loss.item(),
                'vae/reconstruction_loss': reconstruction_loss.item(),
                'vae/quantization_loss': quantization_loss.item(),
                'vae/continuous_loss': continuous_loss.item()
            })
            
            if step > 0 and step % 5000 == 0:
                save_models([self.model], step)
        
        save_models([self.model], num_steps)
            
        


