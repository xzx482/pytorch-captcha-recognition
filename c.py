# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import captcha_setting
from captcha_cnn_model import CNN
from my_dataset import mydataset


from PIL import Image


def c(image):
    #vimage = Variable(image)
 
    image = image.unsqueeze(0)
    vimage = Variable(image)

    predict_label = cnn(vimage)
    c0=captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1=captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2=captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3=captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    pre=c0+c1+c2+c3
    return pre


def 处理图片(image:Image):
    #裁剪1像素的边缘
    image=image.crop((1,1,image.width-1,image.height-1))
    #去除灰色
    imgc=image.convert('RGB')
    h,l=imgc.size
    data=imgc.load()
    for x in range(h):
        for y in range(l):
            p=data[x,y]
            if min(p)>100:
                data[x,y]=(255,255,255,255)
            else:
                data[x,y]=(0,0,0,255)


    imgc.show()
    return imgc


if __name__ == '__main__':
    #'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl',map_location=device))
    print("load cnn net.")
    #'''
    image = Image.open('getVerCode.jfif')
    transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    a=mydataset('dataset\\predict',transform)
    image_=处理图片(image).resize((60,160))
    print(c(transform(image_)))


