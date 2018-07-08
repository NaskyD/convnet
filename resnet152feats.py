import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

pretrained_model = models.resnet152(pretrained=True)

class Resnet152Feats(nn.Module):
  def __init__(self):
    super(Resnet152Feats, self).__init__()
    self.resnet = nn.Sequential(*list(pretrained_model.children())[:-1])
    self.features_pool = list(pretrained_model.features.children())[-1]
    self.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1]) 

  def forward(self, x):
    x = self.features_nopool(x)
    x_pool = self.features_pool(x)
    x_feat = x_pool.view(x_pool.size(0), -1)
    y = self.classifier(x_feat)
    return x_pool, y
