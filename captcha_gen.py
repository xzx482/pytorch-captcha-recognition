# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import captcha_setting
import os

def random_captcha():
    captcha_text = []
    for i in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

image=ImageCaptcha(width=88,height=36,font_sizes=(28,))

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 10000
    path = captcha_setting.TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count//100):
        for j in range(100):
            now = str(int(time.time()))
            text, image = gen_captcha_text_and_image()
            filename = text+'_'+now+'.png'
            image.save(path  + os.path.sep +  filename)
            exit()
        print('saved %d : %s' % (i+1,filename))

