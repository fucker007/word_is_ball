# _*_ coding: utf-8 _*_
import torch
import numpy as np
import torch.nn as nn
class Cognitive(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        print(self.device)

    def forward(self, p, p_label):
        '''
        p: shape (class number * 2)
        p_label: shape (class number, 2)
            
        '''
        #print("p shape: ", p.shape)
        #print("p label shape: ", p_label.shape)
        #print("p array: ", p)
        #print("p label array:", p_label)
        #loss = torch.mean(torch.sub(p, p_label))
        #print("loss: ", loss)
        #loss = torch.mul(loss, loss)
        #loss = torch.sum(loss,2)
        loss = torch.mean(
                    torch.sum(
                        torch.mul(
                            torch.sub(p,p_label),torch.sub(p,p_label)
                            ),1))
        #print("loss : ", loss)
        #return torch.mean(loss).requires_grad_(True)
        return loss
    
    #def backward(self):
    #    return 1.
