#!/usr/bin/env python
# coding: utf-8

# # [Project] 교통 표지판 이미지 분류

# ---

# ## 프로젝트 목표
# - 교통 표지판 이미지 데이터를 분석하고 딥러딩 모델을 통하여 표지판 종류를 예측하는 분류 모델 수행
# - 대량의 이미지 데이터를 전 처리하는 과정과 이에 따른 CNN 모델의 성능 변화를 학습
# ## 프로젝트 목차
# 1. **데이터 분석:** 이미지 데이터를 이루고 있는 요소에 대해서 Dataframe를 사용하여 분석 및 확인
# 
# 2. **데이터 전 처리:** 이미지 데이터를 읽어오고 딥러닝 모델의 입력으로 전 처리
# 
# 3. **딥러닝 모델:** CNN 모델을 구현하고 학습, 평가 및 예측을 수행

# ## 데이터 출처
# -  https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# ## 프로젝트 개요
# 
# 차량 운전을 하면서 도로 교통 표지판을 보고 규칙을 지키는 것은 운전자의 및 교통 안전을 위해서 중요한 일입니다. 만일 사람이 아닌 기계가 이를 수행해야 한다면, 어떻게 표지판을 구분할 수 있을까요? 이러한 물음은 자율 주행차 기술이 발전하면서 중요한 이슈가 되었고, 딥러닝 기술 바탕의 분류 모델이 상당한 수준의 정확도를 보이며 적용되고 있습니다.
# 
# 이번 프로젝트에서는 교통 표지판 분류의 첫 번째 스텝으로 간단하게 교통 표지판 이미지가 입력 되었을 때 이 것이 43 종의 표지판 중 어떤 것인가를 분류하는 딥러닝 모델을 구현합니다. 이를 통하여 교통 표지판 이미지 데이터들의 특징과 CNN 모델을 통하여 분류를 수행하는 것을 학습할 수 있습니다. 

# ---

# ## 학습 모델의 파라미터 범위 입력 

# In[ ]:


# Template을 수행할 parameter들 입력 

# 주요 모델링 argument
opt = 'SGD'
momentum_rate = 0
lr = [0.01, 0.005, 0.001]
start_filters = [32, 64]            # ex) start_filters=16일 경우 첫 block에 들어가는 filter는 filters=[16,16,64]
Res_set_num = [3, 4]                        # Resdual block의 개수 = Res_set의 개수로 정의
identity_L2_num = [[7,8], [7,9]]      # {(각 list의 합+3)*3+2} >= 50 이 성립하도록 숫자 정하기, 대략 13 정도로 맞추면 됨
identity_L3_num = [[3,6,5], [3,7,5]]
identity_L4_num = [[2,3,4,2], [2,3,4,3]]
DO = [0.0, 0.2, 0.4]
CSV_PATH = '/content/drive/MyDrive/CNN_master/오수진/write_test_'+ opt +'.csv'

# 데이터 전처리 과정 필요 번수 
image_size = 32 
Batch_size = 256
valid_size = 0.4
DATA_PATH = '/content/drive/MyDrive/CNN_master/오수진/dataset/'
Tensorboard_PATH = '/content/drive/MyDrive/CNN_master/오수진/result/tensorboard'+'/'+ opt


# ## class Config 정의

# In[ ]:


class Config:
    def __init__(self, opt, lr, start_filters, Res_set_num, identity_L_num, DO=0.0, momentum_rate=0.0, count=0):
        # CNN 모델설정
        self.start_filters = np.array([start_filters, start_filters, start_filters*4])      # 초기에 block(Conv연산 3회. 즉 filter 3개)의 크기 
        self.Res_set_num = Res_set_num                                                      # img_size = 32일 경우, 최대 4set까지 가능(2,3,4 set 활용해볼 예정) >> 
        self.identity_L_num = np.array(identity_L_num)                                     # 각 set에 포함되는 identitiy_block의 개수 
        self.DO = DO                                                                        # DO != 0 일 경우, DropOut을 DO의 크기 비율로 실행
        self.count = count
 
        # 학습을 위한 요소 설정(compile)
        self.opt = opt
        self.momentum_rate = momentum_rate
        self.lr = lr

        # 학습(fit)
        # 최대 EPOCHS = 50이며, callbacks 함수 활용으로 최적치를 찾습니다 
        self.EPOCHS = 1

        # 모델이름
        self.model_name = f' ResNet_{self.opt}{self.momentum_rate}_LR{self.lr}__ResSetNum{self.Res_set_num}_identity_L_num{self.identity_L_num}_model{self.count} '
        self.model_name_csv = ''
        self.make_model_name_csv()

        # accuracy & loss 
        self.train_loss = 0
        self.train_accuracy = 0
        self.val_loss = 0
        self.val_accuracy = 0
        self.test_loss = 0
        self.test_accuracy = 0

        # 변수 출력
        self.print_args()
              
    # 학습 정리할 csv파일 이름 설정
    def make_model_name_csv(self): 
     
        if self.opt == 'Momentum':
            self.model_name_csv = f'ResNet_{self.opt}{self.momentum_rate}_LR{self.lr}__ResSetNum{self.Res_set_num}_identity_L_num{self.identity_L_num}'
        else:
            self.model_name_csv = f'ResNet_{self.opt}_LR{self.lr}__ResSetNum{self.Res_set_num}_identity_L_num{self.identity_L_num}'

    # agr 출력    
    def print_args(self):    
        for key in self.__dict__.keys():
            value = self.__dict__[key]
            print(f'{key}: {value}')
        print('\n')

    # dict 형태로 arg들 반환
    def dict_generator(self):  
        return self.__dict__

    def get_train_result(self, loss, accuarcy):
        self.train_loss = loss
        self.train_accuracy = accuarcy

    def get_val_result(self, loss, accuarcy):
        self.val_loss = loss
        self.val_accuracy = accuarcy

    def get_test_result(self, loss, accuarcy):
        self.test_loss = loss
        self.test_accuracy = accuarcy

    def get_epochs(self, epochs):
        self.EPOCHS = epochs


# ## googledrive mount 및 라이브러리 가져오기(+GPU 할당)

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
import pathlib
import numpy as np    # 행렬연산 
import pandas as pd    # DataFrame 활용 
import matplotlib.pyplot as plt   # 데이터 시각화
import h5py   # .h 활용하여 드라이브에서 낭비되는 loss 줄이기 
from tqdm import tqdm   # for문 지연시간 시각화하여 볼 수 있음
from sklearn.model_selection import train_test_split     # train_test_split: 데이터셋을 나누기 위함
import tensorflow as tf   # tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add, Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D,                                     ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model

get_ipython().run_line_magic('matplotlib', 'inline')

## GPU 사용시
## GPU 메모리 사용 크기만 할당

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


# In[ ]:


import datetime
import csv
get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## 데이터 불러오기, 전처리

# ### 이미지 설정 함수

# In[ ]:


# .csv 파일 불러오기 함수 
df_Meta = pd.read_csv(DATA_PATH + 'Meta.csv')
df_Train = pd.read_csv(DATA_PATH + 'Train.csv')
df_Test = pd.read_csv(DATA_PATH + 'Test.csv')

# .h5파일 불러오기 함수
def load_images_from_h5py(path1, path2):
    h5f = h5py.File(path1, 'r')
    data1 = h5f.get('images')[()]
    h5f.close()

    h5f = h5py.File(path2, 'r')
    data2 = h5f.get('images')[()]
    h5f.close()

    return data1, data2


#.h5 파일을 통한 이미지 데이터 불러오기
path_train_data = DATA_PATH + 'train_images_' +str(image_size) + '.h5'   # train data 경로
path_test_data = DATA_PATH + 'test_images_'+ str(image_size) + '.h5'     # test data 경로

train_images, test_images = load_images_from_h5py(path_train_data, path_test_data)   # train, test image 불러오기

train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, df_Train.ClassId, test_size = valid_size)   # train/validation 나누기

image_height = image_size
image_width = image_size
image_channel = 3      # 컬러 이미지이기에 3채널


# 이미지 데이터 generator 
datagen_kwargs = dict(rescale=1./255)
dataflow_kwargs = dict(batch_size = Batch_size)

# Train
train_datagen = ImageDataGenerator(
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
    datagen_kwargs
)

train_generator = train_datagen.flow(
    train_images, 
    y=train_labels,
    **dataflow_kwargs
)

# Validation
valid_datagen = ImageDataGenerator(datagen_kwargs)
valid_generator = valid_datagen.flow(
    valid_images,
    y=valid_labels,
    shuffle=False, 
    **dataflow_kwargs
)

# Test
test_datagen = ImageDataGenerator(datagen_kwargs)
test_generator = test_datagen.flow(
    test_images, 
    y=df_Test.ClassId, 
    shuffle=False, 
    **dataflow_kwargs
)


# ## ResNet 모델링 

# ### block 및 ResNet50 모델 정의

# In[ ]:


# identity block 함수
def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    # input과 output의 Demension이 같은 경우에 활용 
    x = Conv2D(filters1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    # Skip connect
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    
    return x


# residual block 함수
def residual_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    
    # 입력 Feature Map의 Size(크기): 1/2 배
    # 입력 Feature map의 Dimension(Feature map의 Dimension=filter의 개수): 2배
    # Demention을 맞추기 위해 1*1 Convolution을 사용한다. 
    
    # F(x)
    x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    # block을 거치지 않은 입력 x
    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    # Skip connect: F(x)+x
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


# make_ResNet50_model 함수 
def make_ResNet50_model(start_filters, Res_set_num, identity_L_num, D0):
    shape = (image_width, image_height, image_channel)
    inputs = Input(shape)

    # 초기 layer
    x = Conv2D(64, (3,3), padding='same')(inputs)   
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Res_Set layer 쌓기 
    for i in range(0, Res_set_num):    # i번째 Res_set 
        x = residual_block(x, 3, start_filters)
        for j in range(0, identity_L_num[i]):         # i번째 Res_set에 포함되는 identity_layter의 개수
            x = identity_block(x, 3, start_filters)
        start_filters = start_filters/2

    
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    if DO == True:
        x = Dropout(rate=DO)(x)
    x = Dense(43, activation='softmax')(x)

    model = Model(inputs, x)
    return model        


# ## 객체 생성

# In[ ]:


# 초기화
learing_rate=0
start_filter=[]
Res_set=0
identity_L=[]
Drop=0

# 모든 객체 생성
Config_list = []
count = 0
for learing_rate in lr:
    for start_filter in start_filters:
        for Res_set in Res_set_num:
            if Res_set==2:
                for identity_L in identity_L2_num:
                    for Drop in DO:
                        Config_list.append( Config(opt, learing_rate, start_filter, Res_set, identity_L, Drop, momentum_rate, count))
                        count += 1
            elif Res_set==3:
                for identity_L in identity_L3_num:
                    for Drop in DO:
                        Config_list.append( Config(opt, learing_rate, start_filter, Res_set, identity_L, Drop, momentum_rate, count))
                        count += 1
            elif Res_set==4:
                for identity_L in identity_L4_num:
                    for Drop in DO:
                        Config_list.append( Config(opt, learing_rate, start_filter, Res_set, identity_L, Drop, momentum_rate, count))
                        count += 1
len(Config_list)


# ## Train & Test & save

# In[ ]:


models=[]
summaries=[]


for Config in Config_list[29:30]:
    # 모델 생성
    Config.print_args()
    model = make_ResNet50_model(Config.start_filters, Config.Res_set_num, Config.identity_L_num, Config.DO)
    model.summary()

    # 모델 시각화
    tf.keras.utils.plot_model(model, to_file = '/content/drive/MyDrive/CNN_master/오수진/result/model_img/'+Config.model_name+'.png' , show_shapes=True)

    # optimizer 
    if Config.opt == 'SGD':
        opt=tf.keras.optimizers.SGD(learning_rate=Config.lr)
    elif Config == 'Momentum':
        opt=tf.keras.optimizers.SGD(learning_rate=Config.lr, momentum= Config.momentum_rate )
    elif Config == 'Adagrad':
        opt=tf.keras.optimizers.Adagrad(learning_rate=Config.lr)
    elif Config == 'RMSprop':
        opt=tf.keras.optimizers.RMSprop(learning_rate=Config.lr)
    else:   # optimizer에 대한 설정이 따로 없다면 Adma 사용
        opt=tf.keras.optimizers.Adam(learning_rate=Config.lr)

    # compile
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer = opt,
        metrics=['accuracy']
    )

    # 주소 확인필요
    log_dir = Tensorboard_PATH + '/' + Config.model_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    # callback 객체 생성 (EarlyStopping, ModelCheckpoint)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/CNN_master/오수진/result/checkpoint/' + Config.model_name, mointor='val_accuracy', mode='max',save_best_only=True)
    
    # fit
    history = model.fit(
        train_generator,
        validation_data = valid_generator, # validation 데이터 사용
        epochs = Config.EPOCHS,
        callbacks=[callback, model_checkpoint_callback,tensorboard_callback]
        ) 
    
    # EPOCHS 값 저장 
    Config.get_epochs(len(history.history['val_accuracy']))

    # val_acuuryacy 저장
    idx = np.argmax(history.history['val_accuracy'])
    val_loss = history.history['val_loss'][idx]
    val_accuracy = history.history['val_accuracy'][idx]
    Config.get_val_result(val_loss, val_accuracy)

    # train_accuracy 저장
    train_loss = history.history['loss'][idx]
    train_accuracy = history.history['accuracy'][idx]
    Config.get_val_result(train_loss, train_accuracy)

    # 학습 결과 그래프 저장 
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(Config.EPOCHS)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('/content/drive/MyDrive/CNN_master/오수진/result/train_result_graph/'+Config.model_name+'.png')

    
    # 평가
    test_loss, test_accuracy = model.evaluate(test_generator)
    Config.get_test_result(test_loss, test_accuracy)
    print('test set accuracy: ', test_accuracy)


    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['start_filters','Res_set_num','identity_L_num','DO','count','opt','momentum_rate','lr','EPOCHS','model_name','model_name_csv','train_accuracy', 'test_accuracy', 'test_loss', 'train_loss', 'val_accuracy', 'val_loss'])
        writer.writerow(Config_list[Config.count].__dict__)

    


# ### 텐서보드 여는 코드

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir="/content/drive/MyDrive/CNN_master/오수진/result/tensorboard/SGD"')


# In[ ]:




