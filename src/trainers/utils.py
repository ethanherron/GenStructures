import torch
import os
import wandb


def save_models(models, epoch):
    for model in models:
        # Check if the model is compiled and has the '_orig_mod' attribute
        if hasattr(model, '_orig_mod'):
            orig_model = model._orig_mod
            print(f"Saving original model state_dict for {orig_model.__class__.__name__} at epoch {epoch}")
        else:
            orig_model = model
            print(f"Saving state_dict for {orig_model.__class__.__name__} at epoch {epoch}")
        
        # Save the state_dict of the original model
        torch.save(
            orig_model.state_dict(),
            os.path.join(wandb.run.dir, f'{orig_model.__class__.__name__}_{epoch}.pt')
        )


def load_model(model, weights_path):
    """
    Load the state_dict of a model from a file.
    """
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model
        
        