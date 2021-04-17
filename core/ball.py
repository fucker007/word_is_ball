import torch
import torchvision.models as models
import torch.nn.functional as F

class Ball_Conv(torch.nn.Module):
    def __init__(self, cin, cout, k, activation):
        super(Ball_Conv, self).__init__()
        self.conv1_1 = torch.nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=k,stride=4,padding=4) 
        self.bn1 = torch.nn.BatchNorm1d(cout)
        self.relu_1 = torch.nn.ReLU()
    def forward(self, x):
        #print("input shape: ", x.shape)
        out = self.relu_1(self.bn1(self.conv1_1(x)))
        #print("output shape", out.shape)
        return out

class glob_Conv_49(torch.nn.Module):
    def __init__(self, cin, cout,activation):
        super(glob_Conv_49, self).__init__()
        self.conv1_1 = torch.nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=49,stride=1) 
        self.bn1 = torch.nn.BatchNorm1d(cout)
        self.relu_1 = torch.nn.ReLU()
    def forward(self, x):
        #print("input shape: ", x.shape)
        out = self.relu_1(self.bn1(self.conv1_1(x)))
        #print("output shape", out.shape)
        return out



class Ball_net(torch.nn.Module):
    def __init__(self,num_classes):
        super(Ball_net,self).__init__()
        self.num_classes = num_classes
        self.ball_conv1 = Ball_Conv(cin=9, cout=32, k=9,activation='relu')
        self.ball_conv2 = Ball_Conv(cin=32, cout=64, k=9,activation='relu')
        self.ball_conv3 = Ball_Conv(cin=64, cout=128, k=9,activation='relu')
        self.ball_conv4 = Ball_Conv(cin=128, cout=256, k=9,activation='relu')
        self.ball_conv5 = Ball_Conv(cin=256, cout=512, k=9,activation='relu')
        self.ball_conv6 = glob_Conv_49(cin=512, cout=512,activation='relu')
        #self.dense1 = torch.nn.Linear(512, 128)
        #self.bn1 = torch.nn.BatchNorm1d(128)
        #self.relu_2 = torch.nn.ReLU()
        self.last = torch.nn.Linear(512, num_classes)
        #self.softmax = torch.nn.Softmax(dim=1)

    def freeze(self,model_ft):
        ct = 0
        for child in model_ft.children():
            for param in child.parameters():
                param.requires_grad = False
        return model_ft

    def forward(self,x):
        out = self.ball_conv1(x)
        #print("out 1: ", out.shape)
        out = self.ball_conv2(out)
        #print("out 2: ", out.shape)
        out = self.ball_conv3(out)
        #print("out 3: ", out.shape)
        out = self.ball_conv4(out)
        #print("out 4: ", out.shape)
        out = self.ball_conv5(out)
        #print("out 5: ", out.shape)
        out = self.ball_conv6(out)
        out = out.view(out.size(0), -1)
        out = self.last(out)
        #print("out 6: ", out.shape)
        return out

    def get_model(self):
        return self.last

    def show_model(self):
        print(self.last)




# test Res
if __name__ == "__main__":
    model = models.resnet101(pretrained=True)
    num_classes = 1000
    resnet = Resnet(model, num_classes)
    resnet.show_model()
