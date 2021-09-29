import torch
import torch.nn as nn
import torchvision.models as models

# from efficientnet_pytorch import EfficientNet


def build_model(config, num_class=1):
    # model = BaseModel()
    model = BaseModel2()
    # model = LinerModel()

    return model


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = Conv(7, 32, 3, 1, 1)
        self._multi_conv32 = Conv(32, 32, 1, 1, 0)
        self._down_conv = Conv(32, 64, 3, 2, 1)
        self._multi_conv16 = Conv(64, 64, 1, 1, 0)
        self._fc = nn.Linear(64 * 18, 1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self._conv(x)
        for _ in range(2):
            x = self._multi_conv32(x)
        x = self._down_conv(x)
        for _ in range(2):
            x = self._multi_conv16(x)
        x = x.view(bs, -1)
        x = self._fc(x)
        return x


class BaseModel2(nn.Module):
    def __init__(self):
        super().__init__()
        h = 32
        self._conv = Conv(7, h, 3, 1, 1)
        self._res_block1 = nn.ModuleList([ResidualCSPBlock(h, h) for _ in range(2)])
        self._res_block1 = ResidualCSPBlock(h, h)
        # self._res_block1 = nn.Sequential([ResidualCSPBlock(h, h) for _ in range(2)])
        h2 = h * 2
        self._down_conv = Conv(h, h2, 3, 2, 1)
        self._res_block2 = nn.ModuleList([ResidualCSPBlock(h2, h2) for _ in range(2)])
        self._res_block2 = ResidualCSPBlock(h2, h2)
        # self._res_block2 = nn.Sequential([ResidualCSPBlock(h2, h2) for _ in range(2)])
        self._fc = nn.Sequential(nn.Linear(h2 * 18, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self._conv(x)
        x = self._res_block1(x)
        x = self._down_conv(x)
        x = self._res_block2(x)
        x = x.view(bs, -1)
        x = self._fc(x)

        return x


class ResidualCSPBlock(nn.Module):
    def __init__(self, input_channel, output_channl):
        super().__init__()
        self._init_conv = nn.Conv2d(input_channel, input_channel // 2, 1, 1, 0)
        self._ext = nn.Sequential(
            Conv(input_channel // 2, input_channel // 2),
            Conv(input_channel // 2, input_channel // 2),
        )

    def forward(self, x):
        x = self._init_conv(x)
        x1 = self._ext(x)
        x = torch.cat([x1, x], 1)
        return x
        # return x1 + x


class LinerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc = nn.Sequential(
            nn.Linear(6 * 12, 100),
            # nn.Linear(7 * 6 * 12, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self._linear = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.argmax(x, 1).float()
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self._fc(x)
        x = self._linear(x)
        return x


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(
            input_channel, output_channel, kernel, stride, padding, bias=False
        )
        self._batchnorm = nn.BatchNorm2d(output_channel)
        self._activ = nn.ReLU(inplace=True)

    def forward(self, x):
        # return self._activ(self._conv(x))
        return self._activ(self._batchnorm(self._conv(x)))


if __name__ == "__main__":
    t = torch.rand(1, 7, 12, 6)
    model = BaseModel2()
    print(model(t).shape)
