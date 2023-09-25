import torch
from torch import nn
# import einops

from torchvision.models import * #  convnext_small, ConvNeXt_Small_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext, LayerNorm2d
from torchvision.ops.misc import Conv2dNormActivation
from transformers import ConvNextV2Model
# from transformers import ConvNeXTV2Config, ConvNextV2Model

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
            convnext = convnext_tiny(ConvNeXt_Tiny_Weights.DEFAULT)
        
        elif backbone == 'convnextv1_base':
            convnext = convnext_base(ConvNeXt_Base_Weights.DEFAULT)
        
        config = {
            "sizes": [
                [2, 2],
                [2, 1],
                None,
                [4, 1],
            ]
        }
        
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
            'convnextv2_small',
            'convnextv2_tiny',
            'convnextv2_base'
        ], "{} does not in the pre-defined list".format(backbone)
        backbone = backbone.replace("_", "-")
        convnext = ConvNextV2Model.from_pretrained(f"facebook/{backbone}-22k-224") 
        config = {
            "sizes": [
                [2, 2],
                [2, 1],
                None,
                [4, 1],
            ]
        }
        
        self.base_n_channels = convnext.embeddings.patch_embeddings.out_channels
        self.feature_extractor = convnext
        self.customize(**config)

    @staticmethod
    def down_sampling(in_channels, out_channels, size):
        block = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = size,
            stride = size,
        )
        return block

    def customize(self, sizes):
        self.feature_extractor.embeddings.patch_embeddings = ConvNext_v2.down_sampling(
            in_channels = 3,
            out_channels = self.base_n_channels,
            size = sizes[0],
            groups = out_channels,
        )

        for i in range(1, 4):
            if sizes[i]:
                self.feature_extractor.encoder.stages[i].downsampling_layer = ConvNext_v2.down_sampling(
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
        conv = self.feature_extractor(x).last_hidden_state
        # print(type(conv), conv.shape)
        b, c, h, w = conv.shape

        # conv = einops.rearrange(conv, "b c h w -> w b (c h)")
        conv = conv.reshape(b, c*h, w)
        conv = conv.permute(2, 0, 1)
        # print("rearrange conv: ", conv.shape)
        return conv



    
def convnext_v1_small(stride_pool, kernel_pool, hidden, pretrained, dropout):
    return ConvNext_v1("convnext_v1_small", stride_pool, kernel_pool, hidden, pretrained, dropout)

def test():
    x = torch.rand((1, 3, 32, 128))
    # model = ConvNext_v1('convnext_v1_small',)
    
    model = ConvNext_v1("convnextv1_base")

    summary(model, (3, 32, 128), device = 'cpu')
    # print(model)
    out = model(x)
    print(out.size())


if __name__ == "__main__":
    test()
