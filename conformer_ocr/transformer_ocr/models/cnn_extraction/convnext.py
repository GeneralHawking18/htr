import torch, timm
from torch import nn
# import einops
from PIL import Image
from torchvision.models import * #  convnext_small, ConvNeXt_Small_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext, LayerNorm2d
from torchvision.models.convnext import *
from typing import Any, Callable, List, Optional, Sequence
from torchvision.ops.misc import Conv2dNormActivation
from transformers import ConvNextV2Model
import torchvision.transforms as transforms 
# from transformers import ConvNeXTV2Config, ConvNextV2Model

from functools import partial
from torchsummary import summary
from torch import nn


def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    weights = ConvNeXt_Tiny_Weights.verify(weights)
    # base_channels = 64
    # block_setting = []
    """for i in range(4):
        block_setting.append(CNBlockConfig(base_channels * 2**i, base_channels * 2**(i+1)))"""
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    """block_setting = [
        CNBlockConfig(64, 128, 3),
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 9),
        CNBlockConfig(512, None, 3),
    ]"""
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

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

        assert backbone in [
            'convnextv1_small',
            'convnextv1_tiny',
            'convnextv1_base'
        ], "{} does not in the pre-defined list".format(backbone)

        if backbone == 'convnextv1_small':
            convnext = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
            #for param in convnext.parameters():
                #param.requires_grad = False

        elif backbone == 'convnextv1_tiny':
            convnext = convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)
        
        elif backbone == 'convnextv1_base':
            convnext = convnext_base(ConvNeXt_Base_Weights.DEFAULT)
        config = {
            "sizes": [
                [2, 2], # [4, 4],
                [2, 2],
                [4, 3],
                [4, 1],
            ]
        }
        """config = {
            "sizes": [
                [2, 2], # [4, 4],
                [4, 2],
                [4, 3],
                [4, 1],
            ]
        }"""
        
        self.base_n_channels = convnext.features[0][0].out_channels
        self.feature_extractor = convnext.features
        self.customize(**config)
        
    @staticmethod
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
    
    def customize(self, sizes):
        if sizes[0]:
            self.feature_extractor[0] = ConvNext_v1.down_sampling(
                in_channels = 3,
                out_channels = self.base_n_channels,
                size = sizes[0],
            )

        for i in range(1, 4):
            if sizes[i]:
                self.feature_extractor[i * 2] = ConvNext_v1.down_sampling(
                    in_channels = self.base_n_channels * 2**(i-1),
                    out_channels = self.base_n_channels * 2**(i),
                    size = sizes[i]
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - x: (n, C, H, W)
            - output: (t, n, c)
        """
        conv = self.feature_extractor(x)
        # print(type(conv), conv.shape)
        b, c, h, w = conv.shape

        # conv = einops.rearrange(conv, "b c h w -> w b (c h)")
        conv = conv.reshape(b, c*h, w)
        conv = conv.permute(2, 0, 1)
        # print("rearrange conv: ", conv.shape)
        return conv
    


class ConvNext_v2(nn.Module):
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

        assert backbone in [
            'convnextv2_femto',
            'convnextv2_pico',
            'convnextv2_tiny',
            'convnextv2_small',
            'convnextv2_base',
            
        ], "{} does not in the pre-defined list".format(backbone)
        # backbone = backbone.replace("_", "-")
        convnext = timm.create_model(f'{backbone}.fcmae_ft_in1k', pretrained=True)
        
        
        """print(model)
        
        if backbone == "convnextv2-pico":
            convnext = ConvNextV2Model.from_pretrained("facebook/convnextv2-pico-1k-224") 
        else:
            convnext = ConvNextV2Model.from_pretrained(f"facebook/{backbone}-22k-224")"""
        
        config = {
            "sizes": [
                [2, 2], # [4, 4],
                [2, 2],
                [4, 2],
                [4, 1],
            ]
        }
        
        self.base_n_channels = convnext.stem[0].out_channels
        self.feature_extractor = convnext
        self.customize(**config)

    @staticmethod
    def down_sampling(in_channels, out_channels, size, groups = 1):
        conv_block = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = size,
            stride = size,
            groups = groups,
        )
        norm_block = LayerNorm2d(out_channels, eps=1e-06, elementwise_affine=True)
        conv_block = nn.Sequential(
            conv_block,
            norm_block,
        )
        return conv_block

    def customize(self, sizes):
        self.feature_extractor.stem[0] = ConvNext_v2.down_sampling(
            in_channels = 3,
            out_channels = self.base_n_channels,
            size = sizes[0],
        )

        for i in range(1, 4):
            if sizes[i]:
                self.feature_extractor.stages[i].downsample[1]  = ConvNext_v2.down_sampling(
                    in_channels = self.base_n_channels * 2**(i-1),
                    out_channels = self.base_n_channels * 2**(i),
                    size = sizes[i],
                    # groups = self.base_n_channels * 2**(i),
                )
                
        self.feature_extractor.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - x: (n, C, H, W)
            - output: (t, n, c)
        """
        x = self.feature_extractor(x) #  .last_hidden_state
        # print(type(conv), conv.shape)
        b, c, h, w = x.shape

        # conv = einops.rearrange(conv, "b c h w -> w b (c h)")
        x = x.reshape(b, c*h, w)
        x = x.permute(2, 0, 1)
        # print("rearrange conv: ", conv.shape)
        return x



    
def convnext_v1_small(stride_pool, kernel_pool, hidden, pretrained, dropout):
    return ConvNext_v1("convnext_v1_small", stride_pool, kernel_pool, hidden, pretrained, dropout)

def test():
    backbone = "convnextv2_pico"
    convnext = timm.create_model(f'{backbone}.fcmae_ft_in1k', pretrained=False)
    
    # print(convnext_tiny())
    # img = Image.open("/kaggle/working/preprocess_dataset/267/26.jpg")
    # transform = transforms.ToTensor()
    img = torch.randn(1, 3, 64, 1024)
    # print(convnext)
    # img = transform(img).unsqueeze(0)
    print("img: ", img.shape)
    model = ConvNext_v2("convnextv2_femto")
    # print(model(img).shape)
    
    summary(model, (3, 64, 1024), device = 'cuda', depth = 5)
    # print(model)
    # out = model(x)
    # print(out.size())


if __name__ == "__main__":
    test()
