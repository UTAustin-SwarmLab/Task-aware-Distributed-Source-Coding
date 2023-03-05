import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    "if inc is not outc" -> "if inc != outc"
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        output = torch.tanh(output)
        
        return output


class ImageBasedRLAgent(nn.Module):
    def __init__(self, arch="dist", cdim=3, zdim=512, channels=(32, 32, 32, 32), image_size=256, view=2):
        super(ImageBasedRLAgent, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.arch = arch
        cc = channels[0]
        self.view = view
        self.main = nn.ModuleList()
        for i in range(self.view):
            self.main.append(nn.Sequential(
                nn.Conv2d(cdim, cc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            ))

            sz = image_size // 2
            for ch in channels[1:]:
                self.main[i].add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
                self.main[i].add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
                cc, sz = ch, sz // 2

            self.main[i].add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))

        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = self.view * torch.zeros(self.conv_output_size).view(-1).shape[0]
        # print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        self.fc = nn.Sequential(nn.Linear(num_fc_features, 50*zdim), nn.ReLU(), 
                nn.Linear(50*zdim, 50*zdim), nn.ReLU(), 
                nn.Linear(50*zdim, zdim))
        return

    def calc_conv_output_size(self):
        dim_2 = self.image_size
        dummy_input = torch.zeros(1, self.cdim, dim_2, self.image_size)
        dummy_input = self.main[0](dummy_input)
        return dummy_input[0].shape

    def forward(self, obs):
        if self.view == 2:
            x1, x2 = obs[:, 0:3], obs[:, 3:6]
            y1 = self.main[0](x1).view(x1.size(0), -1)
            y2 = self.main[1](x2).view(x2.size(0), -1)
            y = self.fc(torch.cat((y1, y2), 1))
        else:
            y = self.main[0](obs).view(obs.size(0), -1)
            y = self.fc(y)

        return y
