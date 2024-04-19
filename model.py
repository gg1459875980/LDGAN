import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d(in_channels, out_channels, kernel_size, stride, padding=1, norm=True, activation=True):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if norm:
        layers.append(nn.InstanceNorm3d(out_channels))
    if activation:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            conv3d(in_features, in_features, kernel_size=3, stride=1),
            conv3d(in_features, in_features, kernel_size=3, stride=1, activation=False)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoding layers
        self.encoder = nn.Sequential(
            conv3d(1, 4, kernel_size=7, stride=1, padding=3), 
            conv3d(4, 16, kernel_size=3, stride=2), 
            conv3d(16, 32, kernel_size=3, stride=2),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        
        # Decoding layers
        self.decoder = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose3d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layers for prediction
        self.dense_real = nn.Sequential(
            nn.Linear(in_features=64*64*64*32, out_features=128), 
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        
        self.dense_fake = nn.Sequential(
            nn.Linear(in_features=64*64*64*32, out_features=128), 
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # Encode
        encoding = self.encoder(x)
        
        # Decode
        fake_image = self.decoder(encoding)
        
        # Flatten and predict
        flattened = self.flatten(encoding)
        real_score = self.dense_real(flattened)
        fake_score = self.dense_fake(flattened)
        
        return fake_image, real_score, fake_score

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            conv3d(1, 16, kernel_size=3, stride=2),
            conv3d(16, 32, kernel_size=3, stride=2),
            conv3d(32, 64, kernel_size=3, stride=2),
            conv3d(64, 128, kernel_size=3, stride=1),
            conv3d(128, 1, kernel_size=3, stride=1, norm=False, activation=False),
            nn.Flatten(),
            nn.Linear(in_features=64*64*64*32, out_features=64), 
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Instantiate the models
generator = Generator()
discriminator = Discriminator()
