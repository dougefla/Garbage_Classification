import glob, os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np
import time
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self,model_type='alexnet',label_num = 6,model_save_path = 'results/net.h5'):
        self.model_type = model_type
        self.label_num = label_num
        self.model = self.get_model()
        self.model_save_path = model_save_path
        self.result = None
    # 建立模型结构
    def get_model(self):
        if self.model_type == 'alexnet':
            self.model_savepath = 'results/Alexnet.h5'
            return self.Alexnet()
        else:
            return None
    # 训练模型
    def train(self,train_generator, validation_generator,epochs):

        self.result = self.model.fit_generator(
                # 一个生成器或 Sequence 对象的实例
                generator=train_generator,
                # epochs: 整数，数据的迭代总轮数。
                epochs=epochs,
                # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                steps_per_epoch=2076 // 32,
                # 验证集
                validation_data=validation_generator,
                # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                validation_steps=231 // 32)
        
        self.model.save(self.model_save_path)
        
    def Alexnet(self):
        
        model = Sequential()
        # 第一层
        model.add(Conv2D(filters=96, kernel_size=(11, 11),
                        strides=(4, 4), padding='valid',
                        input_shape=(227, 227, 3),
                        activation='relu'))

        model.add(MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='valid'))
        # 第二层
        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                        strides=(1, 1), padding='same',
                        activation='relu'))

        model.add(MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='valid'))
        # 第三层
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
        # 第四层
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
        # 第五层
        model.add(Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(1, 1), padding='same',
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2), padding='valid'))
        # 第六段
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # 第七层
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # 第八层
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(self.label_num))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
        return model
    
    def plot_training_history(self):
        if self.result == None:
            print("No Result.")
            return

        # 绘制模型训练过程的损失和平均损失
        # 绘制模型训练过程的损失值曲线，标签是 loss
        plt.plot(self.result.history['loss'], label='loss')
        
        # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
        plt.plot(self.result.history['val_loss'], label='val_loss')
        
        # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
        plt.legend(loc='upper right')
        
        # 展示图片
        plt.savefig("results/loss.png")

        # 绘制模型训练过程中的的准确率和平均准确率
        # 绘制模型训练过程中的准确率曲线，标签是 acc
        plt.plot(self.result.history['accuracy'], label='accuracy')
        
        # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
        plt.plot(self.result.history['val_accuracy'], label='val_accuracy')
        
        # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
        plt.legend()
        
        # 展示图片
        plt.savefig("results/accuracy.png")
