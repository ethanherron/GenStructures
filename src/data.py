import os
import torch
import numpy as np
from torch.utils import data


class TopoDataset3D(data.Dataset):
    '''
    Dataset class containing 3D Topology Optimization (SIMP) data.
    Data is stored in npz files for each sample.
    Each file contains an ndarray of shape (3, 64, 64, 64)
    The 3 channels are the initial density, the initial strain energy density, and the final density.
    '''
    def __init__(self, data_path, mode='train'):
        if mode == 'train':
            self.samples = os.path.join(data_path, 'train')
            
        elif mode == 'validation':
            self.samples = os.path.join(data_path, 'validation')
            
        self.list_IDs = os.listdir(os.path.join(self.samples))

    def __len__(self):
        return len(self.list_IDs)

    def log_normalization(self, x):
        '''
        Applies custom log scaling to strain energy data:
        1. Clamps to prevent underflow (1e-22 lower bound (ANSYS solver precision))
        2. Normalizes by max value
        3. Log-transforms and linearly scales to [0,1] range
        '''
        x = torch.clamp(x, min = 1e-22, max = None)
        x = (22 + torch.log10(torch.clamp(x/torch.max(x), 1e-22, 1.0)))/22.0
        return x

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.samples, str(self.list_IDs[index])))['arr_0']
        initial_strain_energy = self.log_normalization(torch.from_numpy(sample[1])).unsqueeze(0)
        final_density = torch.from_numpy(sample[2]).unsqueeze(0)
        return initial_strain_energy, final_density
    
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Specify the data path (adjust as necessary)
    data_path = '.'

    # Instantiate the dataset in training mode
    dataset = TopoDataset3D(data_path, mode='validation')

    # Create a DataLoader with a batch size of 2 for testing
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Iterate over the dataloader, process one batch, and then break
    for batch in dataloader:
        initial_strain_energy, final_density = batch
        print(initial_strain_energy.shape)
        print(final_density.shape)
        print(initial_strain_energy.dtype)
        print(final_density.dtype)
        break
