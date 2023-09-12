import torch
from torch import nn
from timm.models import vgg
import einops
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext, LayerNorm2d
from torchvision.ops.misc import Conv2dNormActivation

from functools import partial
from torchsummary import summary
from torch import nn 

class ConvNext_v1(nn.Module):
    def __init__(
        self, 
        backbone: str = None,
        stride_pool: list = None,
        kernel_pool: list = None,
        d_model: int = None,
        pretrained: bool = None,
        dropout_p: float = None,
    ):
        super().__init__()

        assert backbone in ['convnext_v1_small'], "{} does not in the pre-defined list".format(backbone)

        if backbone == 'convnext_v1_small':
            convnext = convnext_small(ConvNeXt_Small_Weights)
        
            convnext.features[0] = down_sampling(size = 2, in_channels = 3, out_channels = 96)
            convnext.features[2] = down_sampling(size = [2, 1], in_channels = 96, out_channels = 192)
            convnext.features[6] = down_sampling(size = [4, 1], in_channels = 384, out_channels = 768)
            self.feature_extractor = convnext.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - x: (n, C, H, W)
            - output: (t, n, c)
        """
        conv = self.feature_extractor(x)
        conv = einops.rearrange(conv, "b c h w -> w b (c h)")
        
        return conv


def down_sampling(size, in_channels, out_channels):
    block = Conv2dNormActivation(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = size,
        stride = size,
        norm_layer = partial(LayerNorm2d, eps=1e-6), # LayerNorm2d(96),
        activation_layer = None,
    )
    return block
def convnext_v1_small(stride_pool, kernel_pool, hidden, pretrained, dropout):
    return ConvNext_v1("convnext_v1_small", stride_pool, kernel_pool, hidden, pretrained, dropout)

def test():
    x = torch.rand((1, 3, 32, 80))
    model = ConvNext_v1('convnext_v1_small',)
    summary(model, (3, 32, 128), device = 'cpu')
    # print(model)
    out = model(x)
    print(out.size())


if __name__ == "__main__":
    test()
