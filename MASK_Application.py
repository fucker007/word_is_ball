# _*_ coding:utf-8 _*_ 
import os 
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('./')
from data.ALLDATA import ALLDATA 
from data.TEST import TestData
from core.mobilefacenet import MobileFacenet
from core.ball import Ball_net
from core.loss import Cognitive
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score
class Mask:
    def __init__(self, args):
        self.args = args
        self.prefix_dir = os.path.join("./models","Res100_no_finetune_"+args.train_id)
        self.gpus = args.gpus
        self.device = self.find_device()
        self.model = self.get_model()
        self.Datasets = self.get_datasets(args)
        self.Loss = self.get_lossFunction()
        self.opt = self.get_optim()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min",patience=100,min_lr=0.00001,verbose=True)
        t = time.localtime()
        self.writer = SummaryWriter(
                log_dir=os.path.join(self.prefix_dir, "log_tensorboard","%s_%s_%s" % (str(t.tm_mon), str(t.tm_mday), str(t.tm_hour))))
        print("what is scheduler look like: ", self.scheduler)

    def get_datasets(self,args):
        train_set = ALLDATA(
                root_path = args.data_path,
                w = args.input_size,
                h = args.input_size,
                device=self.device) # w , h can changge , mode design
        valid_set = TestData(
                csv_path = args.val_path,
                w = args.input_size,
                h = args.input_size) # w , h can changge , mode design
        #length = len(data)
        #train_size, validate_size = int(0.8*length)+1, int(0.2*length)
        #if length % 5 == 0:
        #    train_size, validate_size = int(0.8*length), int(0.2*length)
        #train_set, valid_set = torch.utils.data.random_split(data, [train_size, validate_size])
        train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        return [train_loader, valid_loader]

    def get_model(self):
        if self.args.pretrain:
            model_path = os.path.join(self.prefix_dir,"model_"+str(self.args.pretrain_epoch)+".pth")
            model = (torch.load(model_path))
        else:
            model = Ball_net(self.args.num_class)
        if len(self.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
        model.to(self.device)
        return model

    def find_device(self):
        print("we wanna get device in computer ! ")
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        #device = torch.device("cpu")
        print(device)
        return device
    
    def get_lossFunction(self):
        # add others loss function on here,Now, just use jiaochashang
        loss = nn.CrossEntropyLoss()
        loss.to(self.device)
        #loss = Cognitive(self.device)
        return loss

    def get_optim(self):
        #opt = optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()),lr = self.args.lr)
        opt = optim.Adam(self.model.parameters(),lr = self.args.lr)
        return opt

    def save_log(self, tags, values, n_iter):
        #dir_path = os.path.join('logs',dir_name)
        for tag, value in zip(tags, values):
            self.writer.add_scalar(tag, value, n_iter)
            self.writer.add_text(tag, str(value), n_iter)

    def compare_numpy(self, a, b):
        #print(a.shape, b.shape)
        # print("[", a[0].detach().cpu().numpy()," | ", b[0].detach().cpu().numpy(),end="]")
        #print(a.shape[0])
        c = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            loss = torch.pow((a[i][0] - b[i][0]),2) + torch.pow((a[i][1] - b[i][1]),2)
            if  loss < 4:
                c[i] = True
            else:
                c[i] = False
        return c
    
    def backwark_grid(self):
        for name, params in self.model.named_parameters():	
            print(name, params)
            if type(params) == "NoneType":
                print(name)
                print("params is None! ")
                break
            print('-->name:', name, '-->grad_requirs:',params.requires_grad,'-->grad_value:',params.grad, "--> data", params.data)
        print("----------------------------------------------------------------------------------------------")
    
    def calculat_roc(self, net_output, label):
        #print(net_output.shape)
        y_pre = torch.argmax(net_output, dim=1)
        acc = (y_pre==label).sum().float() / len(label)
        acc = acc.detach().cpu().numpy()
        #print("predict: ", y_pre)
        #print("label: ", label)
        #print("[", predict_good[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(),end="]")
        #tp = np.sum(np.logical_and(y_pre,np.logical_not(label )))
        #fn = np.sum(np.logical_and(np.logical_not(y_pre), label))
        #tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        #fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        #acc = float(tp)/len(label)
        #print("tpr",tpr, "fpr",fpr,"acc",acc)
        return 0, 0, acc


    def update(self, data):
        img, label = data['ball'].to(self.device), data['label'].to(self.device)
        #print("img shape :", img.shape)
        self.opt.zero_grad()
        output = self.model(img)
        loss = self.Loss(output, label)
        loss.backward()
        # check the info of  backward 
        #self.backwark_grid()
        self.opt.step()
        lr = self.opt.state_dict()['param_groups'][0]['lr']
        #print("learning rate :", lr)
        tpr, fpr, acc = self.calculat_roc(output, label)
        return [loss.cpu().detach().numpy(), tpr, fpr, acc, lr]
        # return [loss.cpu().detach().numpy(),0,0,0,lr]

    def forward(self, data):
        img, label = data['ball'].to(self.device), data['label'].to(self.device)
        #zeros = torch.zeros(label.shape[0], self.args.num_class, 2, device=self.device,dtype=float)
        #onehot_label = torch.nn.functional.one_hot(label, num_classes=self.args.num_class)
        #zeros[:,:,0] = onehot_label
        #zeros[:,:,1] = onehot_label
        output = self.model(img)
        loss = self.Loss(output, zeros)
        tpr, fpr, acc = self.calculat_roc(output, label)
        return [loss.cpu().detach().numpy(), tpr, fpr, acc]

    def save_model(self,epoch):
        torch.save(self.model.state_dict(),
                os.path.join(self.prefix_dir,"model_"+str(epoch)+".pth"))

    def validation(self, valid_loader,epoch):
        val_loss,val_tpr,val_fpr, val_acc = 0.,0.,0.,0.
        step = 0
        for v_batch_i, v_data in enumerate(valid_loader):
            v_values = self.forward(v_data)
            val_loss += v_values[0]
            val_tpr += v_values[1]
            val_fpr += v_values[2]
            val_acc += v_values[3]
            step += 1
        all_rate = [val_loss / step, val_tpr / step, val_fpr/ step, val_acc / step]
        self.save_log(['validation/loss','validation/tpr','validation/fpr','validation/acc'], all_rate, epoch)
        print("[validation]: epoch[%d],---> loss[%.4f], acc[%.4f]" % (epoch, all_rate[0], all_rate[-1]))
        if epoch % 10 == 0:
            self.save_model(epoch)
        return all_rate[0]

    def train(self):
        train_loader, valid_loader = self.Datasets
        for epoch in range(self.args.epochs):
            train_loss, train_tpr, train_fpr, train_acc = 0.,0.,0.,0.
            step = 0
            for batch_i, data in enumerate(train_loader):
                values = self.update(data)
                train_loss += values[0]
                train_tpr += values[1]
                train_fpr += values[2]
                train_acc += values[3]
                step += 1
                if batch_i % 10 == 0:
                    print("[train]: epoch[%d],---> loss[%.4f], acc[%.4f]" % (epoch, values[0], values[-2]))
            all_rate = [train_loss / step, train_tpr / step, train_fpr / step, train_acc / step, values[-1]]
            self.save_log(['train/loss','train/tpr','train/fpr','train/acc','train/learning_rate'], all_rate, epoch)
            self.save_model(epoch)
            # validation
            #average_loss = self.validation(valid_loader,epoch)
            #self.scheduler.step(average_loss)
