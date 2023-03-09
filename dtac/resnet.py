import torch.nn as nn


class LNBlock(nn.Module):
    """
    A residual block with layer normalization. The feature shapes are held constant.
    """

    def __init__(self, feature_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_shape[0], feature_shape[0], kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(feature_shape)
        self.conv2 = nn.Conv2d(feature_shape[0], feature_shape[0], kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(feature_shape)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.ln1(y)
        y = nn.functional.relu(y)
        y = self.conv2(y)

        y += identity
        y = self.ln2(y)
        y = nn.functional.relu(y)
        return y
