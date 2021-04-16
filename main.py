#!/home/liuyang/anaconda3/envs/pytorch/bin/python
# _*_ coding: utf-8 _*_
import argparse
import sys
from train import train 
from test import test
from predict import predict
sys.path.append('./')
from MASK_Application import Mask
def main(args):
    if args.classifer == 'train':
        #print('into train prosses')
        #train(args)
        mask = Mask(args)
        mask.train()
    elif args.classifer == 'test':
        print('into test prosses')
        test(args)
    elif args.classifer == 'predict':
        print('into predict prosses')
        predict(args)

    else:
        print('No good define for parser, check your command please!')

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face quality classfier, for bad face_img and good face_img')
    # train prosess
    parser.add_argument('--train_id', type=str, default="1")
    parser.add_argument('--classifer', type=str, default='train',help='three choose: train: for train net \ntest: for test net\n data: data prosessing ')
    parser.add_argument('--mode_path', type=str, default='./models/')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--input_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--num_class', type=int, default=1000)
    parser.add_argument('--gpus', type=list, default=[0])

    # data prosess
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--val_path', type=str, default='/home/liuyang/DataPublic/train/val.txt')
    parser.add_argument('--version', type=str, default='version 1.0')

    #model prosess
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_epoch', type=int, default=0)



    args = parser.parse_args()
    print(args)
    main(args)

