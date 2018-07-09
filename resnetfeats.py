import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from resnets import resnet101

pretrained_model = resnet101(pretrained=True)

class Resnet101Feats(nn.Module):
  def __init__(self):
    super(Resnet101Feats, self).__init__()
    self.features_nopool = nn.Sequential(*list(pretrained_model.children())[:-2])
    self.features_pool = list(pretrained_model.children())[-2]
    self.classifier = nn.Sequential(list(pretrained_model.children())[-1]) # add one extra fc layer?
    #self.classifier = nn.Sequential(pretrained_mdoel.fc)
  def forward(self, x):
    print("x dimensions: " + str(x.dim()))
    print("x size: ")
    print(x.size())
    x = self.features_nopool(x)
    x_pool = self.features_pool(x)
    x_feat = x_pool.view(x_pool.size(0), -1) #
    print("x_feat size: ")
    print(x_feat.size())
    y = self.classifier(x_feat)
    print("y size: ")
    print(y.size())
    return x_pool, y
