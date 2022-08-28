import torch


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


def load_current_state(filename):
    pass
