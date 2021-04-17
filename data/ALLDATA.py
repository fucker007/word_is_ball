from __future__ import print_function,division
import os
import torch
import cv2
import imgaug.augmenters as iaa
import pandas as pd 
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")



class ALLDATA(Dataset):
    '''
    Achor: liuyang 
        
    '''
    def __init__(self,root_path,w,h,device):
        """
            
        """
        self.device = device
        self.root_path = root_path
        self.landmark = pd.read_csv(
                #os.path.join(root_path,"10_class_train_label_shuffle.csv"),
                #os.path.join(root_path,"train_shuffle.csv"),
                os.path.join(root_path,"train.csv"),
                error_bad_lines=False)
        self.w = w
        self.h = h
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.aug = iaa.Sequential([
                iaa.Resize({"height":self.h,
                            "width":self.w
                    })
            ])

    def __len__(self):
        return len(self.landmark)

    def normalization(self, data):
       _range = torch.max(abs(data))
       return data / _range


    def x_index(self, shape):
        matrix = torch.zeros(shape)
        for i in range(shape[1]):
            matrix[:,i,:] = i+1
        #print(matrix)
        return matrix

    def y_index(self, shape):
        matrix = torch.zeros(shape)
        for i in range(shape[1]):
            matrix[:,:,i] = i+1
        return matrix


    def ball_img(self, img):
        shape = img.shape

        r = (img**2 + self.x_index(shape)**2 + self.y_index(shape)**2)**0.5
        
        alth = torch.acos(img/r)
        
        seta = torch.atan(self.x_index(img.shape) / self.y_index(img.shape))
        
        idx = torch.randperm(shape[2]*shape[1])
        r = self.normalization(r)
        r = r.reshape(shape[0],shape[1]*shape[2])
        r = r[:,idx]
        alth = alth.reshape(shape[0],shape[1]*shape[2])
        alth = alth[:,idx]
        seta = seta.reshape(shape[0],shape[1]*shape[2])
        seta = seta[:,idx]
        return torch.cat((r, alth, seta))

        


    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(
                self.root_path,
                self.landmark.iloc[idx,0]
                ) 
        img = cv2.imread(img_path)
        assert img is not None, f'img is None {img_path}'
        if len(img.shape) == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) 
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        assert 0 not in img.shape, f'0 can not be included in image.shape {img.shape} {img_path}'
        image = self.aug.augment_image(img)
        image = self.transforms(image)
        ball = self.ball_img(image)
        label = self.landmark.iloc[idx,1]
        sample = {'ball':ball,'path':img_path,'label': int(label)}
        return sample
if __name__ == "__main__":
    dt = ALLDATA(root_path="/home/liuyang/DataPublic/train/",
                              w=224,
                              h=224)
    dl = DataLoader(dt,batch_size=1,shuffle=True,num_workers=1)
    for batch_i,data in enumerate(dl):
        print(data['ball'].shape)
        break

        ''''
        print("image path --->:" ,data['path'])
        image = data['image'].numpy()[0]
        image = image.transpose(1,2,0)
        #image = 255 * np.array(image).astype('uint8')
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        print("shape ->>",image.shape,"  label ->>",data['label'].numpy()[0])
        text = str(data['label'].numpy()[0])
        cv2.putText(image, text,(56,56),cv2.FONT_HERSHEY_PLAIN, 2.0,(0,0,225),2)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
