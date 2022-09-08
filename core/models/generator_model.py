import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Block which represents residual connection
    with 2-x convolution layers, instance normalization and relu activation between it.
    Instance normalization and relu activation on outputs is applied.
    ...
    Attributes
    ----------
    instance_norm: torch.nn.modules.instancenorm.InstanceNorm2d
        Layer which applies normalization on each channel separately for each object in sample(batch)
    relu: torch.nn.modules.activation.ReLU
        Applies ReLU activation
    conv_block: torch.nn.modules.container.Sequential
        Block which contains convolutional1 -> instance norm -> relu -> convolutional2
    ...
    Methods
    -------
    forward:
        Forward pass of a model
    """

    def __init__(self, n_channels: int):
        super(ResidualBlock, self).__init__()
        # self.instance_norm = nn.InstanceNorm2d(n_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv_block = nn.Sequential(*[
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(n_channels),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(n_channels),
        ])

    def forward(self, x: torch.Tensor):
        output = self.conv_block(x) + x
        return output


class Generator(nn.Module):
    """
    Convolutional u-net-like generator model.
    It has downsample part of 3 convolutional layers, n_residuals residual layers and
    upsample part of 2 transposed convolutional layers and 1 convolutional layer.
    ...
    Attributes
    ----------
    reflection_pad: torch.nn.modules.padding.ReflectionPad2d
        Makes reflection pad by 3 on each side
    relu: torch.nn.modules.activation.ReLU
        Applies ReLU activation
    pixel_shuffle: torch.nn.modules.pixelshuffle.PixelShuffle
        Rearrange pixels by upscale factor == 2, output: (*, channel / factor ** 2, height * factor, width * factor)
    gathered_layers: list
        All the layers in generator model
    generator_block: torch.nn.modules.container.Sequential
        All the layers in generator model wrapped in torch.nn.Sequential instance, used in forward method
    ...
    Methods
    -------
    gather_layers:
        Makes list of all layers in generator, it is used to create generator block of model
    forward:
        Forward pass of a model
    """

    def __init__(self,
                 in_n_channels: int,
                 hidden_n_channels: int,
                 n_residuals: int = 9):
        super(Generator, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(3)
        self.pixel_shuffle = nn.PixelShuffle(2)  # not used for now
        self.relu = nn.ReLU(inplace=True)
        self.gathered_layers = self.gather_layers(in_n_channels, hidden_n_channels, n_residuals)
        self.generator_block = nn.Sequential(*self.gathered_layers)

    def forward(self, x: torch.Tensor):
        output = self.generator_block(x)
        return output

    def gather_layers(self,
                      in_n_channels: int,
                      hidden_n_channels: int,
                      n_residuals: int):
        layers = []
        # shape: bc * 3 * 256 * 256
        layers += [
            self.reflection_pad,
            # shape: bc * 3 * 262 * 262
            nn.Conv2d(
                in_channels=in_n_channels,
                out_channels=hidden_n_channels,  # for example 64
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            # shape: bc * hc * 256 * 256
            nn.InstanceNorm2d(hidden_n_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=hidden_n_channels,
                out_channels=2 * hidden_n_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            # shape: bc * 2hc * 128 * 128
            nn.InstanceNorm2d(2 * hidden_n_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=2 * hidden_n_channels,
                out_channels=4 * hidden_n_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            # shape: bc * 4hc * 64 * 64
            nn.InstanceNorm2d(4 * hidden_n_channels),
            nn.ReLU(inplace=True),
        ]

        for _ in range(n_residuals):
            layers.append(ResidualBlock(4 * hidden_n_channels))
            # shape: bc * 4hc * 64 * 64

        layers += [
            nn.ConvTranspose2d(
                in_channels=4 * hidden_n_channels,
                out_channels=2 * hidden_n_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # # shape: bc * 8hc * 64 * 64
            # self.pixel_shuffle,
            # shape: bc * 2hc * 128 * 128
            nn.InstanceNorm2d(2 * hidden_n_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=2 * hidden_n_channels,
                out_channels=hidden_n_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # # shape: bc * 4hc * 128 * 128
            # self.pixel_shuffle,
            # shape: bc * hc * 256 * 256
            nn.InstanceNorm2d(hidden_n_channels),
            nn.ReLU(inplace=True),

            # shape: bc * hc * 262 * 262
            nn.Conv2d(
                in_channels=hidden_n_channels,
                out_channels=in_n_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect"),  # The same as using nn.ReflectionPad2d(3) as layer
            # shape: bc * 3 * 256 * 256
            nn.Tanh()
        ]
        return layers


def test_generator() -> None:
    x = torch.rand(2, 3, 256, 256)
    generator = Generator(3, 64)
    output = generator(x)
    print("Output shape:", output.shape)
    return None


if __name__ == "__main__":
    test_generator()
