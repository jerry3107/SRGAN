"""
這個程式是用來建立SRGAN模型並進行訓練的程式
使用的主要套件為tensorflow
使用的資料集為MIRFLICKR資料集
"""


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dropout, Conv2D, Dense, LeakyReLU, Input, Reshape, Flatten, Conv2DTranspose, BatchNormalization, PReLU, Concatenate, add
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import tensorflow.keras.backend as K
from pathlib import Path

def process_path(file_path):
    """
    資料前處理
    """
    file_path = file_path.decode('utf-8')  # 解碼檔案路徑
    try:
        # 載入圖片並將圖片轉為(128,128,3)的大小，並將圖片的數值範圍從[0,255]轉為[-1,1]的範圍
        image = cv2.resize(cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (128, 128))
        image = (np.array(image).astype(np.float32) - 127.5) / 127.5

        # 將圖片縮小，並將圖片的數值範圍從[-1,1]轉為[0,1]的範圍
        small_img = cv2.resize(image, (32, 32))
        small_img = ((np.array(small_img).astype(np.float32) + 1) * 127.5) / 255

        return small_img, image
    except Exception as e:
        print(f"Warning: Could not load image at {file_path} due to {e}")
        return np.array([]), np.array([])  # 如果載入失敗，則回傳空值

def load_data(batch_size=16, file_path='./testdata'):
    """
    載入資料集
    """
    # 載入資料集
    list_ds = tf.data.Dataset.list_files(str(Path(file_path)/'*'), shuffle=False)

    # 資料前處理
    list_ds = list_ds.map(lambda x: tf.numpy_function(process_path, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 將空值的資料刪除
    list_ds = list_ds.filter(lambda x, y: tf.reduce_all([x is not None, y is not None]))

    # 將資料打亂並分批次
    ds = list_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds

"""||請注意cv2.imwrite後面的檔案位置請自行調整||"""
def show_images(images, index = -1):
    """
    展示並保存圖片
    """
    # 將圖片的數值範圍從[-1,1]或[0,1]轉為[0,255]的範圍
    images = np.array(images).astype(np.float32)
    if index == -1:
        images = images*255
    else:
        images = images*127.5+127.5
    # 將圖片的顏色通道由BGR轉為RGB
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    #儲存第i個epoch的圖片
    cv2.imwrite(f"C:\\Users\\srganresult\\test_{index}.png", images)

"""||請注意plt.savefig後面的檔案位置請自行調整||"""
def plot_and_save(loss_real, loss_fake, loss_generate, i):
    """
    繪製並保存loss圖
    """
    # 繪製過去10個EPOCH的loss圖
    time = np.linspace(i-9, i, 10)
    plt.plot(time,loss_real,"r", label='loss_real')
    plt.plot(time,loss_fake,"b", label='loss_fake')
    plt.plot(time,loss_generate,"g", label='loss_generate')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss_real', 'loss_fake', 'loss_generate'], loc='upper left')

    # 保存第i個epoch圖片
    plt.savefig(f"C:\\Users\\srganresult\\testloss_plot_{i}.png")
    plt.close()

# 圖片的shape
Lr_shape = (32,32,3)
Hr_shape = (128,128,3)

def res_block(inputs):
    """
    residual block
    """
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    return add([x, inputs])

def build_G():
    """
    構建生成器
    """
    #輸入層
    conv1 = Sequential([
        Conv2D(64, 9, padding='same'),
        PReLU()
        ]
    )
    #殘差層後的卷積層
    conv2 = Sequential([
        Conv2D(64, 3, padding='same'),
        BatchNormalization()
        ]
    )
    #輸出層
    conv3 = Sequential([
        Conv2D(3, 9, padding='same')
        ]
    )
    #放大兩倍
    up_sampling1 = Sequential([
        Conv2D(64, 3, padding='same'),
        PReLU()
        ]
    )    
    up_sampling2 = Sequential([
        Conv2D(64, 3, padding='same'),
        PReLU()
        ]
    )    

    image_input = Input(Lr_shape) 
    output_conv1 = conv1(image_input)
    
   
     #使用16層residual block
    for i in range(16):
        output_block = res_block(output_conv1)
   
    output_conv2 = conv2(output_block)
    #將圖片放大4倍
    output_up_sampling1  = tf.nn.depth_to_space(up_sampling1(output_conv2),block_size=2, data_format="NHWC") 
    output_up_sampling2  = tf.nn.depth_to_space(up_sampling2(output_up_sampling1),block_size=2, data_format="NHWC")
    output_conv3 = conv3(output_up_sampling2)
    model = Model(inputs=image_input, outputs=output_conv3)
    
    return model

G = build_G() 
#G = load_model('Srgan_Generator'), 重新訓練時請將此行註解掉

def build_D():
    """
    構建判别器
    """
    model = Sequential()
    
    # 卷積層
    model.add(Conv2D(64, 3, input_shape = Hr_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, 3, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 3, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, 3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, 3, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, 3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, 3, strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(512, 3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
   
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 0.00001, beta_1 = 0.9))
    
    return model

D = build_D()
#D = load_model('Srgan_Discriminator'), 重新訓練時請將此行註解掉

def preproces_vgg(x):
    """
    生成器的損失函數
    使用VGG19的block2_conv2層提取特徵
    分別對真實圖片與生成圖片提取特徵
    計算兩者的MSE差值做為content loss
    """
    # 將圖片從 [-1,1] 轉為[0, 255]
    x += 1.
    x *= 127.5
    
    # RGB -> BGR
    x = x[..., ::-1]
    # apply Imagenet preprocessing : BGR mean
    mean = [103.939, 116.778, 123.68]
    _IMAGENET_MEAN = K.constant(-np.array(mean))
    x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))
    
    return x

tf.config.run_functions_eagerly(True)

def vgg_loss(y_true, y_pred):
    # 載入VGG19模型
    vgg19 = VGG19(include_top=False,
                  input_shape=Hr_shape , 
                  weights='imagenet')
    
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    
    features_extractor = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)
    # 將圖片從 [-1,1] 轉為[0, 255]
    features_pred = features_extractor(preproces_vgg(y_pred))
    features_true = features_extractor(preproces_vgg(y_true))
    
    # 計算兩者的MSE差值做為content loss
    return 0.006*K.mean(K.square(features_pred - features_true), axis=-1)

def build_gan():
    """
    生成器及判別器整合為GAN網路
    使用binary_crossentropy做為adversarial loss
    """
    # 暫停判别器，也就是在訓練的時候只優化G的網路權重，而對D保持不變
    D.trainable = False
    # GAN網路的輸入
    gan_input = Input(Lr_shape)
    # GAN網路的輸出
    gan_out = D(G(gan_input))
    # 構建網路
    
    gan = Model(inputs=gan_input, outputs= [G(gan_input),gan_out])
    # 編譯GAN網路，使用Adam優化器，以及加上交叉熵損失函数（一般用於二分類）
    gan.compile(loss= [vgg_loss, "binary_crossentropy"],
                      loss_weights=[1., 1e-3],  optimizer=Adam(learning_rate = 0.00001, beta_1 = 0.9))
    gan.summary()
    return gan

GAN = build_gan()

def smooth_pos_labels(y):
    """
    使true label的值的範圍為[0.8,1]
    """
    return y - (np.random.random(y.shape) * 0.2)

def smooth_neg_labels(y):
    """
    使fake label的值的範圍為[0.0,0.3]
    """
    return y + np.random.random(y.shape) * 0.3

def load_batch(data, batch_size,index=0):
    """
    按批次加載圖片
    """
    return data[index * batch_size: (index+1) * batch_size]

def train(epochs, batch_size, ds):
    """
    訓練函數
    """
    # 生成器損失
    generator_loss = 0
    # img_dataset.shape[0] / batch_size 代表這個數據可以分為幾個批次進行訓練
    
    loss_real = []
    loss_fake = []
    loss_generate = []
       
    for i in range(epochs):
        for j, (LR, HR) in enumerate(ds):
            LR = tf.cast(LR, tf.float32)
            HR = tf.cast(HR, tf.float32)
            x_size = LR.shape[0] 
            # 按批次加載數據
            x = load_batch(LR, x_size)
            y = load_batch(HR, x_size)
            
            # G網路產生圖片
            generated_images = G.predict(x)
            # 產生為1的標籤
            y_real = np.ones(x_size)
            # 將1標籤的範圍變成[0.8 , 1.0]
            y_real = smooth_pos_labels(y_real)
            # 產生为0的標籤
            y_fake = np.zeros(x_size)
            # 將0標籤的範圍變成[0.0 , 0.3]
            y_fake = smooth_neg_labels(y_fake)
            # 訓練真圖片loss
            d_loss_real = D.train_on_batch(y, y_real)
            # 訓練假圖片loss
            d_loss_fake = D.train_on_batch(generated_images, y_fake)
            # 產生為1的標籤
            y_real = np.ones(x_size)
            # 訓練GAN網路，input = fake_img ,label = 1
            generator_loss = GAN.train_on_batch(x,[y,y_real])
            
            print(f'[{j}]. Discriminator real_img : {d_loss_real}. Discriminator fake_img : {d_loss_fake}. Generator_loss: {generator_loss[2]}.')
        loss_real.append(d_loss_real)
        loss_fake.append(d_loss_fake)
        loss_generate.append(generator_loss[2])
        print(f'[Epoch {i}]. Discriminator real_img : {d_loss_real}. Discriminator fake_img : {d_loss_fake}. Generator_loss: {generator_loss[2]}.')

        # 每個epoch保存一次。
        if i%1 == 0:
            # 使用G網路生成1張圖偏
            m = np.random.randint(np.shape(LR)[0])
            print (np.shape(LR[m:m+1]))
            test_images = G.predict(LR[m:m+1])
            print (np.shape(test_images))
            # show 預測 img
            show_images(test_images[0], i)
            # 保存模型
            G.save('Srgan_Generator')
            D.save('Srgan_Discriminator')
        
        if (i+1)%10 == 0:
            # 每10個epoch繪製一次loss圖
            plot_and_save(loss_real, loss_fake, loss_generate, i)
            loss_real.clear()
            loss_fake.clear()
            loss_generate.clear()
        
if __name__ == '__main__':
    """
    進行訓練
    """
    
    train_img_path = "./testdata"# 圖片的位置
    ds = load_data(batch_size=16, file_path=train_img_path)  # This already includes shuffling
    # 顯示數據集的圖片
    for i, (LR, HR) in enumerate(ds):
        show_images(LR.numpy()[0] )
        show_images(HR.numpy()[0], -2)
        if i == 0:
            break

    epochs = 100000
    train(epochs, batch_size = 16, ds =ds)







