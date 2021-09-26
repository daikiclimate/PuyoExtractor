import torch
import torch.nn as nn
import torchvision.models as models

# from efficientnet_pytorch import EfficientNet


def build_model(config, num_class=1):
    model = BaseModel()
    # model_name = config.model
    # if model_name == "mobv2":
    #     model = models.mobilenet_v2(pretrained=False)
    #     in_feature = 1280
    #     model.classifier = nn.Linear(in_feature, num_class)
    # elif model_name == "mobv3_small":
    #     model = models.mobilenet_v3_small(pretrained=False)
    #     in_feature = 576
    #     model.classifier = nn.Linear(in_feature, num_class)
    # elif model_name == "mobv3_large":
    #     model = models.mobilenet_v3_large(pretrained=False)
    #     in_feature = 960
    #     model.classifier = nn.Linear(in_feature, num_class)
    # elif model_name == "vgg19_bn":
    #     model = models.vgg19_bn(pretrained=False)
    #     model.classifier = nn.Sequential(
    #         nn.Linear(in_features=25088, out_features=num_class, bias=True),
    #     )
    # elif model_name == "vgg11_bn":
    #     model = models.vgg11_bn(pretrained=False)
    #     model.classifier = nn.Sequential(
    #         nn.Linear(in_features=25088, out_features=num_class, bias=True),
    #         # nn.ReLU(inplace=True),
    #         # nn.Linear(in_features=4096, out_features=num_class, bias=True)
    #     )
    # elif model_name == "efficientnet":
    #     model_type = model_name + config.model_type
    #     model = EfficientNet.from_name(model_type, num_classes=num_class)

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


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self._conv = nn.Conv2d(
            input_channel, output_channel, kernel, stride, padding, bias=False
        )
        self._batchnorm = nn.BatchNorm2d(output_channel)
        self._activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self._activ(self._batchnorm(self._conv(x)))


if __name__ == "__main__":
    t = torch.rand(1, 7, 12, 6)
    model = BaseModel()
    print(model(t).shape)
