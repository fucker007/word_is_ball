import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
from torchsummary import summary
sys.path.append('./')
# design by conv1 noraml
class Face_quality_Net(nn.Module):
    def __init__(self):
        super(Face_quality_Net,self).__init__()
        '''
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(112,112))  #110 * 110
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3)) #108 * 108
        self.maxpool1 = nn.MaxPool2d((2,2), padding=(1,1))                        #55 * 55  
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3))  #53 * 53 
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3))  #51 * 53
        self.conv5 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3))  #52 * 52
        self.maxpool2 = nn.MaxPool2d((2,2), padding=(1,1))                        #26 * 26
        '''
        self.conv1 = nn.Conv2d(3,128,3)  #111 * 111
        self.conv2 = nn.Conv2d(128,64,3) #110 * 110 
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d((2,2), padding=(1,1))                        #55 * 55  
        self.conv3 = nn.Conv2d(64,32,3,padding=(1,1))  #54 * 54 
        self.conv4 = nn.Conv2d(32,16,3)  #53 * 53
        self.bn4 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d((2,2), padding=(1,1))                        #26 * 26
        self.conv5 = nn.Conv2d(16,16,3)  #52 * 52
        # view
        self.fc1 = nn.Linear(16*5*5,32)
        self.fc2 = nn.Linear(32,2)



    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(self.bn2(self.conv2(out)),2)
        out = F.relu(out) 
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = F.max_pool2d(self.bn4(self.conv4(out)),2)
        out = F.relu(out)
        out = self.maxpool2(out)
        out = self.conv5(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out




class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=stride, padding=1, groups=planes, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        #print("< ----Block---- >\n",out)
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=5):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, 
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.softmax(out,dim=1)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,112,112)
    y = net(x)
    #summary(net,(3,112,112))
    print("y size : -->",y.size(),"\nnet like : \n",net)

if __name__ == '__main__':
    test()
