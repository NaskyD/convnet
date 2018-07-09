import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from resnets import resnet101

pretrained_model = resnet101(pretrained=True)

class Resnet101Feats(nn.Module):
  def __init__(self):
    super(Resnet101Feats, self).__init__()
    self.features_nopool = nn.Sequential(*list(pretrained_model.modules())[:-2])
    self.features_pool = list(pretrained_model.modules())[-2]
    self.classifier = nn.Sequential(list(pretrained_model.modules())[-1]) # add one extra fc layer?

  def forward(self, x):
    print("x dimensions: " + str(x.dim()))
    x = self.features_nopool(x)
    x_pool = self.features_pool(x)
    x_feat = x_pool.view(x_pool.size(0), -1) #
    y = self.classifier(x_feat)
    return x_pool, y
