import torch
import torch.nn as nn
import torchvision


import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
sys.path.append('./')
from data.ALLDATA import ALLDATA 
from core.mobilefacenet import MobileFacenet
from sklearn.metrics import balanced_accuracy_score
from torchsummary import summary
# for can run
def train(args):
    USE_CUDA = torch.cuda.is_available()
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    # define device
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    #val_device = torch.device("cuda:3" if USE_CUDA else "cpu")
    # define net
    #net = MobileNetV2(num_classes=2)
    net = MobileFacenet()
    net = torch.nn.DataParallel(net, device_ids=[0,1,2])
    net.to(device)
    print(net)
    print(3,args.input_size,args.input_size)
    #summary(net,(32,3,args.input_size,args.input_size))
    # define loss
    weights = torch.tensor([35159.,90894.])
    weights.to(device)
    #Loss = nn.CrossEntropyLoss(weight=weights)
    Loss = nn.CrossEntropyLoss()
    Loss.to(device)

    # define optimizer
    opt = optim.Adam(net.parameters(),lr=args.lr)
    
    #define data
    data = ALLDATA(root_path=args.data_path,
                              w=args.input_size,
                              h=args.input_size)
    # split into train and valid
    length = len(data)
    print("length ->>",length)
    train_size,validate_size = int(0.8*length) + 1,int(0.2*length)
    train_set, valid_set = torch.utils.data.random_split(data,[train_size,validate_size])

    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)

    # plot running 
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []
    plt.figure(12)


    for epoch in range(args.epochs):
        running_loss = 0.0
        running_precision =0.0 
        total_step = 0
        for i,data in enumerate(train_loader):
            img, label = data['image'].to(device), data['label'].to(device)
            #print("image shape -->",img.shape)
            #print("label shape -->",label.shape,"eg ->> ", label[0])
            # optimizer
            opt.zero_grad()
            # loss
            output = net(img)
            #print("output shape -->",output.shape,"eg ->>", output[0:5])
            loss = Loss(output,label)
            #loss = BCE(output,label)
            loss.backward()
            opt.step()
            #print("loss item --> ",loss.item())
            running_loss += loss.item()*args.batch_size
            total_step += args.batch_size
            # precision AP
            y_pre = torch.argmax(output,dim=1).cpu()
            precision = balanced_accuracy_score(label.cpu(),y_pre)
            print("predict: ",y_pre)
            print("label: ", label.cpu())
            #running_precision.append(precision)
            running_precision += precision
            total_train_loss = running_loss / total_step
            if i % 20 == 0:
                print('[epoch: %d, batch id: %5d]---> precision: %.5f, loss: %.5f' % (epoch + 1, i + 1, precision, running_loss / total_step))
        # validation data for echo epoch 
        validation_loss = 0.0
        validation_precision = 0.0
        validation_step = 0
        for j,val_data in enumerate(valid_loader):
            img, label = val_data['image'].to(device), val_data['label'].to(device)
            output = net(img)
            loss = Loss(output,label)
            # loss = F.nll_loss(output,label)
            validation_loss += loss.item()*args.batch_size
            validation_step += args.batch_size
            y_pre = torch.argmax(output,dim=1).cpu()
            precision = balanced_accuracy_score(label.cpu(),y_pre)
            validation_precision += precision
            #validation_precision.append(precision)
            total_val_loss = validation_loss / validation_step
            if j % 15 == 0:
                print('[!validation like: epoch: %d]---> precision: %.5f, loss: %.5f' % (epoch + 1, precision, validation_loss / validation_step))

        # plot running train acc and val acc

        import numpy as np
        acc_train.append(running_precision/int(length*0.8 / args.batch_size))
        acc_val.append(validation_precision/int(length*0.2 / args.batch_size))

        loss_train.append(total_train_loss)
        loss_val.append(total_val_loss)


        x_data = np.arange(epoch+1)
        print("x_data ->>>", x_data, " acc_train ->>> ", acc_train, "acc_val ->>>> ", acc_val) 
        plt.cla()
        plt.subplot(211)
        plt.plot(x_data,acc_train,"r-",label="train acc")
        plt.plot(x_data,acc_val,"b-", label="val acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.pause(0.1)
        
        plt.subplot(212)
        plt.plot(x_data,loss_train,"r-",label="train loss")
        plt.plot(x_data,loss_val,"b-", label="val loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.pause(0.1)


        # save model after 50 epoch
        if epoch % 10 == 0:
            torch.save(net.module.state_dict(),args.mode_path+"/model_"+args.train_id+"_"+str(epoch)+".pth")
    torch.save(net.module.state_dict(),args.mode_path+"/model_last"+args.train_id+".pth")
# for test train code 
if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    # define net
    net = Face_quality_Net().cuda(0)
    summary(net,(3,112,112))
    # define loss

    # define optimizer
    opt = optim.Adam(net.parameters(),lr=0.0001)
    
    #define data
    data = fqd(csv_path="./data/face_quality_label_0_1_classfer.txt",
                              w=112,
                              h=112)
    # split into train and valid
    length = len(data)
    print("length ->>",length)
    train_size,validate_size = int(0.8*length),int(0.2*length)
    train_set, valid_set = torch.utils.data.random_split(data,[train_size,validate_size])

    train_loader = DataLoader(train_set,batch_size=128,shuffle=True,num_workers=5)
    valid_loader = DataLoader(valid_set,batch_size=128,shuffle=True,num_workers=5)


    for epoch in range(1000):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            img, label = data['image'].cuda(0), data['label'].cuda(0)
            #print("image shape -->",img.shape)
            #print("label shape -->",label.shape,"eg ->> ", label[0])
            # optimizer
            opt.zero_grad()
            # loss
            output = net(img)
            print("output shape -->",output.shape,"eg ->>", output[0])
            loss = F.nll_loss(output,label)
            #loss = BCE(output,label)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            # precision AP
            y_pre = torch.argmax(output,dim=1)
            precision = balanced_accuracy_score(label,y_pre)
            if i % 20 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
    PATH = './face_quality_net.pth'
    torch.save(net.state_dict(),PATH)
        






