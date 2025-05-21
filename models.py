import torch
from torch import nn


class VGGNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=62, hidden_dim=16):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(
                hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(
                hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim * 8, hidden_dim * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 4 * 4, hidden_dim * 8 * 4 * 2),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 4 * 2, hidden_dim * 8 * 4 * 2),
            nn.ReLU(inplace=True),
        )

        self.fc_top = nn.Linear(hidden_dim * 8 * 4 * 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_top(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.kaiming_uniform_(
            self.conv1.weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_uniform_(
            self.conv2.weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_uniform_(
            self.conv_bypass.weight.data, mode="fan_in", nonlinearity="relu"
        )

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
        )
        # self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)
        if out_channels != in_channels:
            self.bypass = nn.Sequential(nn.Upsample(scale_factor=2), self.conv_bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Reconstructor(nn.Module):
    def __init__(self, z_dim, gen_size, out_channels, out_res=32):
        super(Reconstructor, self).__init__()
        self.z_dim = z_dim
        self.gen_size = gen_size
        self.out_channels = out_channels
        self.out_res = out_res
        assert out_res in [32, 64], "Output resolution can be one of 32/64."

        self.convT = nn.ConvTranspose2d(self.z_dim, self.gen_size * 4, 4, stride=1)
        self.final = nn.Conv2d(self.gen_size, self.out_channels, 3, stride=1, padding=1)
        nn.init.kaiming_uniform_(
            self.convT.weight.data, mode="fan_in", nonlinearity="linear"
        )
        nn.init.xavier_uniform_(self.final.weight.data, nn.init.calculate_gain("tanh"))

        if out_res == 32:
            self.model = nn.Sequential(
                ResBlock(self.gen_size * 4, self.gen_size * 2, stride=2),
                ResBlock(self.gen_size * 2, self.gen_size, stride=2),
                ResBlock(self.gen_size, self.gen_size, stride=2),
                nn.BatchNorm2d(self.gen_size),
                nn.ReLU(),
                self.final,
                nn.Tanh(),
            )
        else:
            self.model = nn.Sequential(
                ResBlock(self.gen_size * 4, self.gen_size * 2, stride=2),
                ResBlock(self.gen_size * 2, self.gen_size, stride=2),
                ResBlock(self.gen_size, self.gen_size, stride=2),
                ResBlock(self.gen_size, self.gen_size, stride=2),
                nn.BatchNorm2d(self.gen_size),
                nn.ReLU(),
                self.final,
                nn.Tanh(),
            )

    def forward(self, z):
        z = z.view(len(z), self.z_dim, 1, 1)
        z = self.convT(z)
        return self.model(z)
