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


def load_current_state(d_monet_filename,
                       d_photo_filename,
                       g_monet_filename,
                       g_photo_filename,
                       d_optimizer,
                       g_optimizer,
                       device="cpu"):
    """
    Loads models and optimizers from state dicts
    Parameters
    ----------
    d_monet_filename: path-like object
        Filepath of monet paintings discriminator
    d_photo_filename: path-like object
        Filepath of photo discriminator
    g_monet_filename: path-like object
        Filepath of monet paintings generator
    g_photo_filename: path-like object
        Filepath of photo generator
    device:
        Torch device
    """
    d_monet = Discriminator(3, 64).to(device)
    d_photo = Discriminator(3, 64).to(device)
    g_monet = Generator(3, 64).to(device)
    g_photo = Generator(3, 64).to(device)
    print("Loading states...")
    d_monet.load_state_dict(
        torch.load(d_monet_filename, map_location=torch.device(device))["model_state_dict"]
    )
    d_photo.load_state_dict(
        torch.load(d_photo_filename, map_location=torch.device(device))["model_state_dict"]
    )
    g_monet.load_state_dict(
        torch.load(g_monet_filename, map_location=torch.device(device))["model_state_dict"]
    )
    g_photo.load_state_dict(
        torch.load(g_photo_filename, map_location=torch.device(device))["model_state_dict"]
    )
    d_optimizer = torch.optim.Adam(list(d_monet.parameters()) + list(d_photo.parameters()))
    g_optimizer = torch.optim.Adam(list(g_monet.parameters()) + list(g_photo.parameters()))
    d_optimizer = torch.optim.Optimizer.load_state_dict(
        d_optimizer,
        torch.load(d_monet_filename, map_location=torch.device(device))["optimizer_state_dict"]
    )
    g_optimizer = torch.optim.Optimizer.load_state_dict(
        g_optimizer,
        torch.load(g_monet_filename, map_location=torch.device(device))["optimizer_state_dict"]
    )
    print("Loaded states")
    return d_monet, d_photo, g_monet, g_photo, d_optimizer, g_optimizer


if __name__ == "__main__":
    pass
