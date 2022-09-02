import torch
from torch import nn


class Discriminator(nn.Module):
    """
    Discrimination part of the model.
    ...
    Attributes
    ----------
    main: torch.nn.modules.container.Sequential
        Model stack with all layers
    fc: torch.nn.modules.linear.Linear
        Fully-connected linear layer to classify embedding extracted from convolutions
    flattener: torch.nn.modules.flatten.Flatten
        Makes 4d tensor flat
    sigmoid: torch.nn.Sigmoid
        Applies sigmoid non-linearity on the output
    ...
    Methods
    -------
    forward:
        Forward pass of a model
    """

    def __init__(self,
                 in_n_channels: int,
                 hidden_n_channels: int):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(in_features=30 * 30,
                            out_features=1)
        self.flattener = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        # Input shape: bc * 3 * 256 * 256
        self.main = nn.Sequential(*[
            nn.Conv2d(in_channels=in_n_channels,
                      out_channels=hidden_n_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      # bias=False
                      ),
            # shape: bc * hc * 128 * 128
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=hidden_n_channels,
                      out_channels=2 * hidden_n_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      # bias=False
                      ),
            # shape: bc * 2hc * 64 * 64
            nn.InstanceNorm2d(2 * hidden_n_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=2 * hidden_n_channels,
                      out_channels=4 * hidden_n_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      # bias=False
                      ),
            # shape: bc * 4hc * 32 * 32
            nn.InstanceNorm2d(4 * hidden_n_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=4 * hidden_n_channels,
                      out_channels=8 * hidden_n_channels,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      # bias=False
                      ),
            # shape: bc * 8hc * 31 * 31
            nn.InstanceNorm2d(8 * hidden_n_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=8 * hidden_n_channels,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      # bias=True
                      )
            # shape: bc * 1 * 30 * 30

        ])

    def forward(self, x: torch.Tensor):
        output = self.main(x)
        # output = self.flattener(output)
        # output = self.fc(output)
        # Apply sigmoid activation to the discriminator output then pass it to Cross entropy or MSE loss
        output = self.sigmoid(output)
        return output


def test_discriminator() -> None:
    x = torch.rand(2, 3, 256, 256)
    discriminator = Discriminator(3, 64)
    output = discriminator(x)
    print("Output shape:", output.shape)
    return None


if __name__ == "__main__":
    test_discriminator()
