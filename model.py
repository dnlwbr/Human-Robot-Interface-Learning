import copy
import math

from functools import partial
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection
import torchvision.models.detection.faster_rcnn as faster_rcnn
import torchvision.models.detection.ssd as ssd
from torchvision import models


class NeuralNet(nn.Module):
    def __init__(self, name, backbone, num_classes):
        super(NeuralNet, self).__init__()
        self.model = copy.deepcopy(backbone)

        # Store name to access meta info of default pytorch weights
        self.backbone_name = name

        # Freeze all the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Add background to num_classes
        # Edit: Already added to ImageFolder classes
        # num_classes = num_classes + 1

        # Replace pre-trained head (requires_grad=True by default)
        if isinstance(backbone, torchvision.models.detection.FasterRCNN):
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        elif isinstance(backbone, torchvision.models.detection.FCOS):
            in_channels = self.model.backbone.out_channels
            num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
            self.model.head.classification_head.num_classes = num_classes
            self.model.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            prior_probability = 0.01    # Default value of FCOSClassificationHead
            torch.nn.init.normal_(self.model.head.classification_head.cls_logits.weight, std=0.01)
            torch.nn.init.constant_(self.model.head.classification_head.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))
        elif isinstance(backbone, torchvision.models.detection.RetinaNet):
            in_channels = self.model.backbone.out_channels
            num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
            self.model.head.classification_head.num_classes = num_classes
            self.model.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            prior_probability = 0.01  # Default value of RetinaNetClassificationHead
            torch.nn.init.normal_(self.model.head.classification_head.cls_logits.weight, std=0.01)
            torch.nn.init.constant_(self.model.head.classification_head.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))
        elif name == "ssd300_vgg16":
            anchor_generator = self.model.anchor_generator
            num_anchors = anchor_generator.num_anchors_per_location()
            if hasattr(self.model.backbone, "out_channels"):
                in_channels = self.model.backbone.out_channels
            else:
                in_channels = det_utils.retrieve_out_channels(self.model.backbone, (300, 300))
            if len(in_channels) != len(anchor_generator.aspect_ratios):
                raise ValueError(
                    f"The length of the output channels from the backbone ({len(in_channels)}) do not match the length of the anchor generator aspect ratios ({len(anchor_generator.aspect_ratios)})"
                )
            self.model.head.classification_head = ssd.SSDClassificationHead(in_channels, num_anchors, num_classes)
        elif name == "ssdlite320_mobilenet_v3_large":
            in_channels = det_utils.retrieve_out_channels(self.model.backbone, (320, 320))
            anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
            num_anchors = anchor_generator.num_anchors_per_location()
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
            self.model.head.classification_head = ssd.SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        else:
            raise TypeError("Invalid backbone")
            # raise NotImplementedError

    def forward(self, inputs, targets=None):
        if self.training:
            out = self.model(inputs, targets)
        else:
            out = self.model(inputs)
        return out


if __name__ == "__main__":
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    backbone = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    print(backbone)
    model = NeuralNet(backbone, 2)
    print(model)
