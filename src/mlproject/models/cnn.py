from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, size=64):
        super().__init__()

        conv_kernel_size = 3
        conv_padding = 1
        pool_kernel_size = 2

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size),

            nn.Conv2d(16, 32, kernel_size=conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size),


            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))
