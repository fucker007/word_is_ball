import torch
import torchvision.models as models
import torch.nn.functional as F
class Resnet(torch.nn.Module):
    def __init__(self,model, num_classes):
        super(Resnet,self).__init__()
        self.num_classes = num_classes
        #self.resnet_layer = self.freeze(torch.nn.Sequential(*list(model.children())[:-1]))
        self.resnet_layer = self.freeze(torch.nn.Sequential(*list(model.children())[:-1]))
        self.relu_1 = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(2048, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.relu_2 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(512, 2)
        self.relu_3 = torch.nn.ReLU()
        #self.last = torch.nn.Linear(2048, num_classes * 2)

    def freeze(self,model_ft):
        ct = 0
        for child in model_ft.children():
            for param in child.parameters():
                param.requires_grad = False
        return model_ft


    def Cognitive_layer(self,x):
        # input: x shape is (class number, 2*class number)
        pass

    def forward(self,x):
        x = self.resnet_layer(x)
        #print("x size :",x.size(0))
        print("resnet_layet shape: ", x.shape)
        x = x.view(x.size(0),-1)
        x = self.relu_1(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu_2(x)

        x = self.dense2(x)
        x = self.relu_3(x)
        #print("just 2 layer shape:",x.shape)
        #x = self.last(x)
        #x = x.view(x.size(0),self.num_classes, 2)
        return x

    def get_model(self):
        return self.model

    def show_model(self):
        print(self.resnet_layer)




# test Res
if __name__ == "__main__":
    model = models.resnet101(pretrained=True)
    num_classes = 1000
    resnet = Resnet(model, num_classes)
    resnet.show_model()
