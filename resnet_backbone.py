import torchvision.models.resnet as resnet
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
from torch import Tensor


class ResnetBackbone(resnet.ResNet):
    def __init__(
            self,
            block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        resnet.ResNet.__init__(self, block, layers, num_classes, zero_init_residual, groups, width_per_group,
                               replace_stride_with_dilation, norm_layer)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _resnet(
        arch: str,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> ResnetBackbone:
    model = ResnetBackbone(block, layers, **kwargs)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResnetBackbone:
    return _resnet("resnet50", resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)