import torch.nn as nn
from torchvision.models.resnet import Bottleneck,conv1x1
import torch.nn.functional as F

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, roi_size:int,in_channels, au_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(roi_size//2)
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, au_classes)

    def forward(self, x):
        #pdb.set_trace()
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HeadModule(nn.Module):
    def __init__(self,inplanes=1024,planes=512,blocks=3,groups: int = 1,width_per_group: int = 64):
        super().__init__()
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self._norm_layer = nn.BatchNorm2d
        self.layer = self._make_layer(Bottleneck, planes, blocks,stride=2)
    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self,rois):
        return self.layer(rois)






