import torch
from torch import nn
from transformer_ocr.models.cnn_extraction.vgg import vgg11_bn, vgg19_bn
from transformer_ocr.models.cnn_extraction.convnext import * # convnext_v1_small

class FeatureExtraction(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(FeatureExtraction, self).__init__()

        if model_name == 'vgg11_bn':
            self.model = vgg11_bn(**kwargs)
        elif model_name == 'vgg19_bn':
            self.model = vgg19_bn(**kwargs)
        elif "convnextv1" in model_name:
            self.model = ConvNext_v1(model_name)
        elif "convnextv2" in model_name:
            self.model = ConvNext_v2(model_name)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

