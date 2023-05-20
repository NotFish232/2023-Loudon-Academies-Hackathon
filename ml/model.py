import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self


class Generator(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        # (batch_size, 2048)
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(2048, 15 * 15 * 256),  # (batch_size, 15 * 15 * 256)
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        # (batch_size, 256, 15, 15)
        self.inverse_convolutional_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 9, stride=4),  # (batch_size, 64, 65, 65)
            nn.Dropout2d(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 8, stride=4),  # (batch_size, 32, 264, 264)
            nn.Dropout2d(),
            nn.LeakyReLU(),
        )
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=9),  # (batch_size, 3, 25, 256)
            nn.Dropout2d(),
            nn.Sigmoid(),
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = self.fully_connected_layers(x)
        x = x.view(-1, 256, 15, 15)
        x = self.inverse_convolutional_layers(x)
        x = self.convolutional_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        # (batch_size, 3, 256, 256)
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 64, 5),  # (batch_size, 64, 252, 252)
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(4, 4),  # (batch_size, 64, 63, 63)
            nn.Dropout2d(),
            nn.Conv2d(64, 256, 4),  # (batch_size, 256, 60, 60)
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(4, 4),  # (batch_size, 256, 15, 15)
            nn.Dropout2d(),
            nn.Conv2d(256, 512, 2),  # (batch_size, 512, 14, 14)
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # (batch_size, 512, 7, 7)
            nn.Dropout2d(),
        )
        # (batch_size, 512 * 7 * 7)
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),  # (batch_size, 2048)
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(2048, 256),  # (batch_size, 256)
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 1),  # (batch_size, 1)
            nn.Dropout(),
            nn.Sigmoid(),
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = self.convolutional_layers(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fully_connected_layers(x)
        return x


def main() -> None:
    gen = Generator()
    disc = Discriminator()

    gen.eval()
    disc.eval()

    print(f"Generator: {sum(i.numel() for i in gen.parameters())}")
    print(f"Discriminator: {sum(i.numel() for i in disc.parameters())}")
    with T.no_grad():
        y = gen(T.randn((1, 2048)))
        y_hat = disc(y)
    print(y.shape)
    print(y_hat.shape)


if __name__ == "__main__":
    main()
