#_*_ coding: utf-8 _*_
import os
from torch.utils.data import Dataset, DataLoader
from data.TEST import TestData
from sklearn.metrics import balanced_accuracy_score
from core.mobilefacenet import MobileFacenet
import  torch
import cv2
import numpy as np 
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
def test(model_path,data_path,input_size = 112):
    save_path = "wrong_img"
    #get data
    data = TestData(csv_path=data_path,
                              w=input_size,
                              h=input_size)
    data_loader = DataLoader(data,batch_size=32,num_workers=12)
    # load model
    model = MobileFacenet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    step = 0
    model.eval()
    precision = 0.
    for i, data in enumerate(data_loader):
        cat_img = np.zeros([112,112,3])
        flag = 0
        img, label = data['image'].to(device), data['label'].to(device)
        path = data['path']
        output = model(img)
        y_pre =  torch.argmax(output, dim=1).cpu()
        precision += balanced_accuracy_score(label.cpu(),y_pre)
        print("step: %d -->  acc: %.4f" % (step, precision / step ))
        print(y_pre,"\n", label)
        step += 1
        j =0
        for j  in range(y_pre.shape[0]):
            if int(y_pre[j]) != label[j].cpu().numpy():
                wrong_img = img[j].cpu().numpy()
                wrong_img = wrong_img.transpose(1,2,0) * 255
                wrong_img = cv2.cvtColor(wrong_img,cv2.COLOR_RGB2BGR) 
                cv2.putText(wrong_img,str(label[j].cpu().numpy()),(10, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                cat_img = np.concatenate((cat_img, wrong_img))
                print(path[j])
                flag = 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if flag == 1:
            cv2.imwrite(
                    os.path.join(save_path,str(i)+".jpg"),
                    cat_img)
        cv2.destroyAllWindows()

    print(precision / step)
if __name__ == "__main__":
    test_path = "/ext_disk3/data_liuyang/Mask_dataset/MAFA/test_label.csv"
    model_path = "models/model_5_110.pth"
    test(model_path, test_path)
