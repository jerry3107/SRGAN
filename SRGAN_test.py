"""
這個程式是用來進行結果測試的
輸入圖片路徑和模型路徑，以及選取區域的左上角座標
程式會載入模型，並對圖片進行預測，展示結果
比較原圖、bicubic放大、預測結果
"""
 
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dropout, Conv2D, Dense, LeakyReLU, Input, Reshape, Flatten, Conv2DTranspose,  MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from pathlib import Path


def draw_rectangle(img,x, y, shape ):
    """
    將選取的區域用紅色方框框起來
    """
    left_up = (y, x)
    right_down =  (y+shape, x+shape)
    color = (0, 0, 255) # red
    thickness = 2 # 寬度 (-1 表示填滿)
    cv2.rectangle(img, left_up, right_down, color, thickness) 
 
    return img 

def show_images(HR,SR,LR, x, y, shape):
    """
    展示圖片
    """
    LR = np.array(LR).astype(np.float32)
    HR = np.array(HR).astype(np.float32)
    SR = np.array(SR).astype(np.float32)
    LR = cv2.cvtColor(LR, cv2.COLOR_BGR2RGB)
    bicubic = cv2.resize(LR,(128, 128), interpolation=cv2.INTER_CUBIC)#bicubic放大
    SR = cv2.cvtColor(SR, cv2.COLOR_BGR2RGB)
    SR = (SR+1)*127.5/255
    HR = cv2.cvtColor(HR, cv2.COLOR_BGR2RGB)
    HR = draw_rectangle(HR, x, y, shape)
    #展示圖片
    cv2.imshow("HR", HR)
    cv2.imshow("SR", SR)
    cv2.imshow("LR", LR)
    cv2.imshow("bicubic", bicubic)#展示bicubic放大的圖片
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(model, img, x, y):
    #判斷圖片是否讀取成功
    if img.shape[0] == 0:
        print('圖片讀取失敗')
        return
    elif x >96 or y >96:
        print('X或Y太大')
        return
    elif x<0 or y<0:
        print('X或Y太小')
        return
    #選取區域大小
    shape=32
    #將圖片縮放到128*128
    img = cv2.resize(img,(128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)/255
    #將圖片切成32*32的小圖
    LR = img[:,x:x+shape,y:y+shape]
    #放大
    SR = model.predict(LR[0:1])
    SR = np.array(SR).astype(np.float32)
    #展示圖片
    show_images(img[0],SR[0],LR[0], x, y, shape)

if __name__ == '__main__':
    photo_path = 'C:\\Users\\test.jpg'#圖片路徑
    model_path = 'Srgan_Generator'#模型路徑，這部分我有附在'Srgan_Generator'，可以直接使用
    #選取區域的左上角座標
    x=50
    y=70
    #載入模型
    model = load_model(model_path)
    #讀取圖片
    img = cv2.imread(photo_path,3)
    #預測並展示圖片
    predict(model, img, x, y)
