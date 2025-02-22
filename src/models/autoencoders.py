import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

########################################################
#               Encoder and Decoder                    #
########################################################

class Encoder_3d(nn.Module):
    def __init__(self, in_channels=1, dim=32, n_downsample=3, latent_channels=None):
        super(Encoder_3d, self).__init__()
        layers = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(in_channels, dim, 7),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(0.2),
        ]

        current_dim = dim
        for _ in range(n_downsample):
            layers += [
                nn.Conv3d(current_dim, current_dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm3d(current_dim * 2),
                nn.LeakyReLU(0.2),
            ]
            current_dim *= 2

        if latent_channels is None:
            latent_channels = current_dim
        if latent_channels != current_dim:
            layers.append(nn.Conv3d(current_dim, latent_channels, kernel_size=1))

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model_blocks(x)
        return x

class Decoder_3d(nn.Module):
    def __init__(self, out_channels=1, dim=32, n_upsample=3, latent_channels=None):
        super(Decoder_3d, self).__init__()

        init_channels = dim * 2 ** n_upsample
        if latent_channels is None:
            latent_channels = init_channels

        layers = []
        if latent_channels != init_channels:
            layers.append(nn.Conv3d(latent_channels, init_channels, kernel_size=1))

        current_channels = init_channels
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose3d(current_channels, current_channels // 2, 4, stride=2, padding=1),
                nn.InstanceNorm3d(current_channels // 2),
                nn.LeakyReLU(0.2),
            ]
            current_channels //= 2

        layers += [
            nn.Conv3d(current_channels, current_channels // 2, 3, padding=1), 
            nn.InstanceNorm3d(current_channels // 2),
            nn.LeakyReLU(0.2), 
            nn.Conv3d(current_channels // 2, out_channels, 3, padding=1)
        ]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model_blocks(x)
        return x
    
########################################################
#               Gaussian/Quantization                  #
########################################################

class DiagonalGaussian(nn.Module):
    '''
    Continuous latent space
    '''
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean = z
        if self.sample:
            std = 0.1
            z = mean * (1 + std * torch.randn_like(mean))
            loss = z.pow(2).mean()
            return z, loss
        else:
            return mean
        
class VectorQuantizer(nn.Module):
    """
    A simple vector quantizer for 3D data.
    This implementation quantizes the input volume and computes a commitment loss.
    It assumes inputs of shape (batch, channels, depth, height, width).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        
        z_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        embed_sq = torch.sum(self.embedding.weight ** 2, dim=1).unsqueeze(0)
        z_embed = torch.matmul(z_flat, self.embedding.weight.t())   
        distances = z_sq + embed_sq - 2 * z_embed   
        
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)
        
        loss = torch.mean((z_q.detach() - z_flat) ** 2) + \
               self.beta * torch.mean((z_q - z_flat.detach()) ** 2)
        
        z_q = z_flat + (z_q - z_flat).detach()
        
        z_q = z_q.view(z.shape)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        
        return z_q, loss, encoding_indices

########################################################
#         Multi-Headed Variational Autoencoder         #
########################################################

class MultiHeadedVariationalAutoencoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim: int, n_downsample: int, latent_channels: int, quantizer_beta: float = 1., kl_beta: float = 0.01, inference: bool = False):
        super().__init__()
        self.inference = inference
        self.encoder_strain_energy = Encoder_3d(in_channels, dim, n_downsample, latent_channels // 2)
        self.encoder_density = Encoder_3d(in_channels, dim, n_downsample, latent_channels // 2)
        
        self.decoder = Decoder_3d(out_channels, dim, n_downsample, latent_channels)
        
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=latent_channels, beta=0.25)
        self.gaussian = DiagonalGaussian(sample=not self.inference)
        
        self.quantizer_beta = quantizer_beta
        self.kl_beta = kl_beta
        
    def forward(self, strain_energy: Tensor, density: Tensor) -> Tensor:
        se_latent = self.encoder_strain_energy(strain_energy)
        d_latent = self.encoder_density(density)
        
        se_q, quantization_loss, _ = self.quantizer(se_latent)
        d_z, continuous_loss = self.gaussian(d_latent)
        
        density_reconstruction = self.decoder(torch.cat([se_q, d_z], dim=1))
        
        reconstruction_loss = F.binary_cross_entropy_with_logits(density_reconstruction, density)
        
        total_loss = reconstruction_loss + self.quantizer_beta * quantization_loss + self.kl_beta * continuous_loss
        
        return total_loss, reconstruction_loss, quantization_loss, continuous_loss
    
    @torch.no_grad()
    def encode_se(self, strain_energy: Tensor) -> Tensor:
        se_latent = self.encoder_strain_energy(strain_energy)
        se_q, _, _ = self.quantizer(se_latent)
        return se_q
    
    @torch.no_grad()
    def encode(self, strain_energy: Tensor, density: Tensor) -> Tensor:
        '''
        Returns the latent representation of the strain energy and density.
        Intended only for inference when training the latent diffusion model.
        '''
        se_q = self.encode_se(strain_energy)
        d_z = self.gaussian(self.encoder_density(density))
        return se_q, d_z
    
    @torch.no_grad()
    def decode(self, se_q: Tensor, d_z: Tensor) -> Tensor:
        return self.decoder(torch.cat([se_q, d_z], dim=1))


if __name__ == "__main__":
    # Create a random tensor with shape [batch, 1, 64, 64, 64] (using batch size 2 for example)
    x = torch.randn(2, 1, 64, 64, 64)

    # Initialize the encoder and decoder.
    # (Assuming an Encoder_3d class is defined elsewhere in this file with default parameters)
    encoder = Encoder_3d(latent_channels=16)
    print(f'Encoder param count: {sum(p.numel() for p in encoder.parameters())}')
    decoder = Decoder_3d(latent_channels=16)
    print(f'Decoder param count: {sum(p.numel() for p in decoder.parameters())}')

    # Pass the input through the encoder to obtain the latent representation.
    latent = encoder(x)
    print("Latent shape:", latent.shape)

    # Pass the latent representation through the decoder to reconstruct the input.
    decoded = decoder(latent)
    print("Decoded shape:", decoded.shape)
    
    # Initialize the multi-headed variational autoencoder.
    vae = MultiHeadedVariationalAutoencoder(in_channels=1, out_channels=1, dim=32, n_downsample=3, latent_channels=8)
    print(f'VAE param count: {sum(p.numel() for p in vae.parameters())}')

    # Pass the input through the VAE to obtain the reconstructed input and the loss.
    loss, reconstruction_loss, quantization_loss, continuous_loss = vae(x, x)
    print(f'Loss shape: {loss.shape}')
    print(f'Reconstruction loss shape: {reconstruction_loss.shape}')
    print(f'Quantization loss shape: {quantization_loss.shape}')
    print(f'Continuous loss shape: {continuous_loss.shape}')
    
    se_q = vae.encode(x)
    print(f'SE_q shape: {se_q.shape}')
    se_d = vae.encode(x)
    print(f'SE_d shape: {se_d.shape}')
    x_hat = vae.decode(se_q, se_d)
    print(f'x_hat shape: {x_hat.shape}')
