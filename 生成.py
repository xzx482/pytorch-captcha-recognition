# -*- coding: UTF-8 -*-
from PIL import Image,ImageDraw,ImageFont
import random
import time
import captcha_setting
import os

所有字符="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
画布大小=(88,36)


def 生成验证码图片():
    #创建画布
    image = Image.new('RGB', 画布大小, (255, 255, 255))
    #创建画笔
    draw = ImageDraw.Draw(image)
    字符=''
    开始位置=[画布大小[0]*0.1*random.random(),int(画布大小[1]*0.2*random.random())]
    font=ImageFont.truetype("simsun.ttc",size=28)
    #生成4个字符
    for i in range(4):
        字符_=所有字符[random.randint(0,len(所有字符)-1)]
        字符+=字符_
        draw.text((开始位置[0]+(4*(random.random()-0.5)),开始位置[1]+(4*(random.random()-0.5))),字符_,(0,0,0),font=font,)
        开始位置[0]+=画布大小[0]*0.25
    #随机噪点
    imgc=image.convert('RGB')
    h,l=imgc.size
    data=imgc.load()
    for i in range(500):
        x,y=random.randint(5,h-5),random.randint(5,l-5)#至少要减1
        data[x,y]=(255,255,255)
    return 字符,imgc




if __name__ == '__main__':
    '''
    a=生成验证码图片()
    print(a[0])
    a[1].show()
    exit()
    '''
    count = 10000
    path = captcha_setting.TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count//100):
        for j in range(100):
            now = str(int(time.time()))
            text, image = 生成验证码图片()
            filename = text+'_'+now+'.png'
            image.save(path  + os.path.sep +  filename)
        print(i+1)

