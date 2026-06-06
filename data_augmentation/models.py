from torch import nn
from torchinfo import summary

class Generator(nn.Module):
    """
    Generator model.

    According to the paper, the Generator Block must have the following:
      1) Transposed convolutional layers.
      2) Batchnorm, except for the final layer.
      3) Each batch norm is followed by a ReLU activation.
      4) If its the final layer, we will use Tanh activation after the trasposed convolutional layer.
    """
    def __init__(self, channels=1, hidden_dim=64, z_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            # Input shape : 1, 100,  1, 1 (batch_size, channel, height, width)
            # Output shape : (hidden_dim * 8) x 4 x 4
            nn.ConvTranspose2d(z_dim, hidden_dim * 16, (4, 4), 1, 0),
            nn.BatchNorm2d(hidden_dim * 16),
            nn.ReLU(True),

            # Output shape : (hidden_dim * 4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 16, hidden_dim * 8, (4, 4), 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),

            # Output shape : (hidden_dim * 4) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, (4, 4), 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            # Output shape : (hidden_dim x 2) x 32 x 32
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, (4, 4), 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            # Output shape : (hidden_dim x 2) x 64 x 64
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, (4, 4), 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            # Output shape : hidden_dim x 128 x 128
            nn.ConvTranspose2d(hidden_dim, channels, (4, 4), 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    

# generator = Generator()
# summary(generator, input_size=(1, 100, 1, 1))
    
class Discriminator(nn.Module):
    """
    Discriminator model.
    """
    def __init__(self, channels=1, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            # Input shape : (channels, 128, 128)
            # State shape : (hidden_ dim * 2) x 64 x 64
            nn.Conv2d(channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            # State shape : (hidden_ dim * 2) x 32 x 32 
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State shape : (hidden_ dim * 4) x 16 x 16 
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State shape : (hidden_ dim * 8) x 8 x 8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # State shape : (hidden_ dim * 8) x 8 x 8
            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # Output shape : (1, 1, 1)
            nn.Conv2d(hidden_dim * 16, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
# discriminator = Discriminator()
# summary(discriminator, input_size=(1, 1, 128, 128))