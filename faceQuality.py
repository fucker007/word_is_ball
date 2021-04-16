# _*_ coding:utf-8 _*_
import cv2 
class FaceQuality:
    def __init__():
        pass
    def __init__(self,args):
        self.blu_thresh = [100,args.blu_thresh][args.blu_thresh != 0]
        self.ill_thresh = [0.75,args.ill_thresh][args.ill_thresh != 0]
        print("blurry thresh: ",self.blu_thresh," illum thresh:",self.ill_thresh)

    # quality
    def quality(self,img):
        label = dict(
                ok = 0,
                blurry=0,
                compelet = 0,
                angle = 0,
                illum = 0,
                )
        # any image channel to 1 
        img = self.to_1_cha(img)
        print("image shape --->>",img.shape)
        #
        label['blurry'] = self.is_blurry(img)
        label['compelet'] = self.is_compelet(img)
        label['angle'] = self.is_angle(img)
        label['illum'] = self.is_illum(img)

        if label['blurry'] and label['compelet'] and label['angle'] and label['illum']:
            label['ok'] = 1
        landmark = dict()
        result = dict(
                labels = label,
                landmark = landmark
                )
        return result

    # blurry 1
    def is_blurry(self,img):
        value = cv2.Laplacian(img,cv2.CV_64F).var()
        print("laplacian: ",value)
        return 1 if value < self.blu_thresh else 0

    # blurry 2 
    #energy函数计算 需要再写
    def energy(img):
        '''
        :param img:narray 二维灰度图像
        :return: float 图像越清晰越大
        '''
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0]-1):
            for y in range(0, shape[1]-1):
                out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
        return out

    def is_compelet(self,img):

        return 0
    
    # is_angle
    def is_angle(self,img):

        return 0

    #
    def is_illum(self,img):
        from matplotlib import pyplot as plt
        img_list = plt.hist(img.ravel(),256,[0,256])
        print("img_list --->>",img_list[0].shape[0])
        all_pix = img.shape[0] * img.shape[1]
        qian_val = img_list[0][0:int(256/4)].sum() / all_pix     
        guo_val = img_list[0][int(256*7/8):].sum() / all_pix     
        guo_val_6_8 = img_list[0][int(256*6/8):].sum() / all_pix     

        print("all pix is [",all_pix,"] ","qian val : ",qian_val, " guo_val: ", guo_val)

        return 1 if qian_val > self.ill_thresh or guo_val > 0.3 or guo_val_6_8 > 0.5 else 0

    # channle to 1
    def to_1_cha(self,img):
        if img.shape[0] == 3:
            return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            return img
        else:
            raise AssertionError("img type is not supported in this system")

    
if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='face quality')
    parser.add_argument('--blu_thresh', type=int, default=0)
    parser.add_argument('--ill_thresh', type=int, default=0)
    args = parser.parse_args()

    #
    facequal = FaceQuality(args)

    df = pd.read_csv("../data/face_quality_label_0_1_classfer_version_2.txt")
    for index,row in df.iterrows():
        img = cv2.imread(row['path'])
        result = facequal.quality(img)
        if result['labels']['illum'] == 1:
            cv2.imwrite("./test_data/bao/"+str(index)+".png",img)
            cv2.destroyAllWindows()
        elif result['labels']['blurry'] == 1:
            cv2.imwrite("./test_data/blur/"+str(index)+".png",img)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite("./test_data/good/"+str(index)+".png",img)
            cv2.destroyAllWindows()
        print(result)
