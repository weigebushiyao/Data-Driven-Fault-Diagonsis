import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import RMSprop, SGD
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

print('Data loading...')
train_x_read=np.array(pd.read_csv("train_data.csv"))
train_y_read=np.array(pd.read_csv("train_labels.csv"))
train_y_read.reshape(-1,1)
test_x_read=np.array(pd.read_csv("test_data.csv"))
test_y_read=np.array(pd.read_csv("test_labels.csv"))
test_y_read.reshape(-1,1)
print('Data loading process complete\n')
train_num=train_x_read.shape[0]-62
test_num=test_x_read.shape[0]-62
train_x=np.ndarray((64,25,train_num,1))   #(train_x_read.shape[0]-62)个64*26的数据
train_y=np.ndarray((train_num,2))       #(train_x_read.shape[0]-62)个数据标签
test_x=np.ndarray((64,25,test_num,1))     #(test_x_read.shape[0]-62)个64*26的数据
test_y=np.ndarray((test_num,2))         #(test_x_read.shape[0]-62)个数据标签
print('Transforming training data...')
for i in range(63,train_x_read.shape[0]):
    train_x[:,:,i-63,0]=train_x_read[i-63:i+1,:]
    train_y[i-63,:]=train_y_read[i,:]
train_x=np.transpose(train_x,(2,3,0,1))
print('Transforming test data...')
for i in range(63,test_x_read.shape[0]):
    test_x[:,:,i-63,0]=test_x_read[i-63:i+1,:]
    test_y[i-63,:]=test_y_read[i,:]
test_x=np.transpose(test_x,(2,3,0,1))
print('Transforming process complete\n')
    

"""
定义卷积神经网络结构
"""
print('Model building...')
model = Sequential()
#定义卷积层1(卷积核5*5*32,输入数据1*64*26,输出32*64*26)
model.add(Conv2D(32,(5,5),border_mode='same',dim_ordering='th',input_shape=(1,64,25)))
model.add(Activation('relu'))
#定义池化层1(最大值池化2*2,步长1,输入32*64*26,输出32*32*13)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same'))
#定义遗忘层1
model.add(Dropout(0.25))
#定义卷积层2(卷积核5*5*32,输入数据32*32*13,输出64*32*13)
model.add(Conv2D(64,(5,5),border_mode='same',dim_ordering='th',input_shape=(32,32,13)))
model.add(Activation('relu'))
#定义池化层2(最大值池化2*2,步长1,输入64*32*32,输出64*16*7)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same'))
#定义遗忘层2
model.add(Dropout(0.25))
#定义全连接层1(输入64*14*14=12544,输出4096)
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
#定义全连接层2(输入4096,输出100)
model.add(Dense(100))
model.add(Activation('relu'))
#定义全连接层3(输入100,输出2)
model.add(Dense(2))
model.add(Activation('softmax'))
print('Model building process complete\n')

#编译CNN网络
print('Model compiling...')
model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])
print('Model compiling process complete\n')

#训练模型
print('Model training...')
model.fit(train_x,train_y)
score = model.evaluate(test_x, test_y)
model.save("CNN.model")
plot_model(model, to_file='modelcnn.png',show_shapes=True)
print('Model training process complete\n')
