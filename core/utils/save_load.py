import torch
from ..models import Discriminator, Generator
import os


def save_current_state(model, optimizer, filename):
    """
    Saves current state of a model and an optimizer passed as parameters
    Parameters
    ----------
    model: torch.nn.Module
        Pytorch model which state dict will be saved
    optimizer: torch.optim.Optimizer
        Pytorch optimizer which state dict will be saved
    filename: path-like object
        Filepath of saving
    """
    print("Saving current state...")
    saved_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(saved_state, filename)
    print("Saved current state")


def load_current_state(filename, device="cpu"):
    """
    Loads models and optimizers from state dicts
    Parameters
    ----------
    filename: path-like object
        Filepath of saving
    device:
        Torch device
    """
    d_monet = Discriminator(3, 64).to(device)
    d_photo = Discriminator(3, 64).to(device)
    g_monet = Generator(3, 64).to(device)
    g_photo = Generator(3, 64).to(device)
    print("Loading states...")
    print(os.listdir(filename))
    d_monet.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["model_state_dict"]
    )
    d_photo.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["model_state_dict"]
    )
    g_monet.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["model_state_dict"]
    )
    g_photo.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["model_state_dict"]
    )
    d_optimizer = torch.optim.Optimizer.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["optimizer_state_dict"]
    )
    g_optimizer = torch.optim.Optimizer.load_state_dict(
        torch.load(filename, map_location=torch.device(device))["optimizer_state_dict"]
    )
    print("Loaded states")
    return d_monet, d_photo, g_monet, g_photo, d_optimizer, g_optimizer


if __name__ == "__main__":
    load_current_state("../../states")
