# _*_coding:utf-8 _*_

import cv2
import os
import sys
from shutil import copyfileobj
from data import ALLDATA
from torch.utils.data import Dataset, DataLoader
from tool.is_qianbaoguang import get_mean
from tool.is_mohu import variance_of_laplacian

def save_temp_file(source):
    target_path = os.path.join(os.path.dirname(source),"contine_data_prosess_temp.txt")
    if os.path.exists(target_path):
        if os.path.getsize(target_path) > 10:
            return target_path 
        else:
            print("文件为空: "+'cp '+source+' '+target_path)
            os.system('cp '+source+' '+target_path)
    else:
        # 创建一个文件
        with open(target_path,'a'):
            os.utime(target_path,None)
        try:
            target = open(target_path)
            os.system('cp '+source+' '+target_path)
        except IOError as e:
            print("unable to copy file. %s" % e)
        except:
            print("Unexpected error:", sys.exc_info())
        return target_path
def delete_first_lines(filename, count):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[count:])
    fout.write(b)
    fin.close()
    fout.close()

def prosess_data(args):
    '''
        将数据按照总的流程进行处理，就算由于执行不利，可以继续处理，而不是重头再来。
        并且希望能够加入gpu中进行处理和计算。

    '''
    # 操作之前先备份源文件，生成一个temp文件操作，对文件进行读写。
    path = save_temp_file(args.data_path)
    print("备份文件的路径为： ",path,"size is : ",str(os.path.getsize(path)))
    print("*********************************************************")
    copy_source_path = os.path.join(os.path.abspath(args.data_path),"contine_data_prosess_temp.txt")
    dt = Face_quality_Dateset(csv_path=args.data_path,w=112,h=112)
    dl = DataLoader(dt, batch_size=1, shuffle=True,num_workers=1)

    for batch_i,data in enumerate(dl):
        # source image
        print("data path is -->>",data['path'][0])
        img = cv2.imread(data['path'][0])

        # 为了保证公平性，将图片统一缩放成112之后再计算
        img = cv2.resize(img,(112,112),interpolation=cv2.INTER_AREA)


        # 使用传统的方法检出欠曝光照片
        if args.is_bg:
            mean = get_mean(img)
            if mean < args.bg_th:
                with open('./data/is_bao/face_quality_qianbao.txt', 'a+') as f:
                    f.write(data['path'][0]+","+"4\n")
            else:
                with open('./data/is_bao/face_quality_no_qianbao.txt','a+') as f:
                    f.write(data['path'][0]+","+str(data['label'].numpy()[0])+"\n")

            print("images exposure mean -->> ",mean)
        # 检出模糊的照片
        if args.is_mh:
            fm = variance_of_laplacian(img)
            if fm < args.mh_th:
                with open('./data/is_mohu/face_quality_mohu.txt', 'a+') as f:
                    f.write(data['path'][0]+","+"0\n")
            else:
                with open('./data/is_mohu/face_quality_no_mohu.txt', 'a+') as f:
                    f.write(data['path'][0]+","+str(data['label'].numpy()[0])+"\n")
        # if 图像裁剪的方法放在jupyter中

        # 读取完一个文件,就删除文中的一行
        delete_first_lines(path, count=1)

    print("*********************************************************")
    print("备份文件的路径为： ",path,"size is : ",str(os.path.getsize(path)))
    print("生成新文件路径为： ",path,"size is : is_mohu and is_bao")
