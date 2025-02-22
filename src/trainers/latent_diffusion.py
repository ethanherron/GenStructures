import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from models.autoencoders import MultiHeadedVariationalAutoencoder
from models.dit import DiT
from .utils import save_models

class DDPM(nn.Module):
    def __init__(self, model: DiT, nfe: int):
        super().__init__()
        self.model = model
        self.nfe = nfe
        self.betas = self.linear_beta_schedule()
        self.ddpm_schedules = self.register_ddpm_schedules(self.betas)
        
        for k, v in self.ddpm_schedules.items():
            self.register_buffer(k, v)
    
    def linear_beta_schedule(self):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / self.nfe
        beta_start = scale * 0.0015
        beta_end = scale * 0.0155
        return torch.linspace(beta_start, beta_end, self.nfe, dtype = torch.float32)
    
    def register_ddpm_schedules(self, beta_t):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """

        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  
            "oneover_sqrta": oneover_sqrta,  
            "sqrt_beta_t": sqrt_beta_t,  
            "alphabar_t": alphabar_t,  
            "sqrtab": sqrtab,  
            "sqrtmab": sqrtmab,  
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  
        }
    
    def noise(self, x, t, noise):
        xt = (self.sqrtab[t, None, None, None, None] * x + 
               self.sqrtmab[t, None, None, None, None] * noise)
        return xt
    
    def forward(self, se_q: Tensor, d_z: Tensor) -> Tensor:
        noise = torch.randn_like(d_z)
        t = torch.randint(0, self.nfe, (d_z.shape[0],), device=d_z.device)
        xt = self.noise(d_z, t, noise)
        loss = F.mse_loss(xt, self.model(torch.cat([se_q, xt], dim=1), t / self.nfe))
        return loss
    
    def sample(self, x):
        '''
        x is dummy data to get the shape. can be noise if you know shape
        or data if you're lazy like me
        '''
        x = torch.randn(x.shape).to(x.device)
        for i in tqdm(range(self.nfe - 1, 0, -1)):
            t = torch.full((x.shape[0],), i / self.nfe, device=x.device)
            z = torch.randn_like(x) if i > 1 else 0
            eps = self.model(x, t)
            x = (self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)
        return x
    

class LatentDiffusionTrainer:
    def __init__(self, 
                 dit: DiT, 
                 vae: MultiHeadedVariationalAutoencoder, 
                 train_loader: DataLoader, 
                 device: torch.device, 
                 optimizer: torch.optim.Optimizer, 
                 num_steps: int
                 ):
        self.ddpm = DDPM(dit, num_steps)
        self.vae = vae
        self.vae.inference = True
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.scaler = GradScaler()
        
        self.ddpm.to(self.device)
        self.vae.to(self.device)
    
    def train(self, num_steps: int):
        self.ddpm.model.train()
        self.vae.eval()
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
                se_q, d_z = self.vae.encode(strain_energy, density)
                loss = self.ddpm(se_q, d_z)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            wandb.log({
                'ldm/loss': loss.item()
            })
            
            if step > 0 and step % 5000 == 0:
                save_models([self.ddpm.model], step)
                
        save_models([self.ddpm.model], num_steps)
            
            
        


