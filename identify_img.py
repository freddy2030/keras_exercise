from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras import losses
import keras
import pandas as pd
import cv2
# import os 

# os.environ["PATH"] += os.pathsep + '/Users/freddy/.local/share/virtualenvs/dl_python-xsW6y-xu/lib/python3.6/site-packages/graphviz/'

def simulate_xor():
    x_train = np.array([[1,0], [1,1], [0,0], [0,1]])
    y_train = np.array([[1], [0], [0], [1]])

    model = Sequential()
    # model.add(Conv2D(64, (3, 3), activation = 'relu'))


    model.add(Dense(8, input_dim=2, activation='relu'))# bias_initializer='zeros'))
    # model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD(lr=0.9, momentum=0.0, decay=0.0, nesterov=False))

    hist = model.fit(x_train, y_train, epochs=100)
    plt.scatter(range(len(hist.history['loss'])), hist.history['loss'])

    # plot_model(model, to_file = './model.png')
    model.save("simple.h5")
    # loss_and_metrics = model.evaluate(x_train, y_train)
    # print(loss_and_metrics)
    # x_test = np.array([[1,1]])
    print('test: ', model.predict(x_train))
    # plt.show()

target = cv2.imread('target.png')

model = Sequential()

model.add(Conv2D(3,3,3,input_shape= target.shape ,name='conv_1'))

model.load_weights('identify_target.h5', by_name=True)

target_batch = np.expand_dims(target,axis=0)

conv_target = model.predict(target_batch)

target_img = np.squeeze(conv_target, axis=0)

max_img = np.max(target_img)

min_img = np.min(target_img)

target_img = target_img-(min_img)

target_img=target_img/(max_img - min_img)

target_img = target_img*255

cv2.imwrite('conv1_output.jpg',target_img)

# model.add(Conv2D(3,3,3,input_shape= target.shape,name='conv_1'))#加入一个卷积层，filter数量为3，卷积核size为（3,3）

# model.add(MaxPooling2D(pool_size=(3,3)))#加入一个pooling 层,size为（3,3）

# model.add(Activation('relu'))# 加入激活函数'ReLu', 只保留大于0 的值

# model.add(Conv2D(3,3,3,input_shape= target.shape,name='conv_2'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Activation('relu'))

# model.add(Conv2D(3,3,3,input_shape= target.shape,name='conv_3'))

# model.add(Activation('relu'))

# model.add(Conv2D(3,3,3,input_shape= target.shape,name='conv_4'))

# model.add(Activation('relu'))

# model.add(Flatten())#把上层输出平铺

# model.add(Dense(8, activation='relu',name='dens_1'))

# model.save_weights("identify_target.h5")
