import glob, os

# 数据集路径
data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import time
def train_model(train_generator, validation_generator, save_model_path='results/net.h5',
              log_dir="results/logs/"):
    """
    创建、训练和保存深度学习模型
    :param train_data: 训练集数据
    :param test_data: 测试集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------

    # AlexNet
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
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])


    d = model.fit_generator(
            # 一个生成器或 Sequence 对象的实例
            generator=train_generator,
            # epochs: 整数，数据的迭代总轮数。
            epochs=100,
            # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            steps_per_epoch=2076 // 32,
            # 验证集
            validation_data=validation_generator,
            # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
            validation_steps=231 // 32)
    print("Done!")
    # 模型保存
    model.save(save_model_path)
    print("Model Saved!")

    return d, model

from keras.models import load_model

def load_and_model_prediction(validation_generator):
    """
    加载模型和模型评估，打印验证集的 loss 和准确度
    :param validation_generator: 预测数据
    :return: 
    """
    # 加载模型
    model = load_model('results/net.h5')
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
def plot_training_history(res):
    """
    绘制模型的训练结果
    :param res: 模型的训练结果
    :return:
    """
    # 绘制模型训练过程的损失和平均损失
    # 绘制模型训练过程的损失值曲线，标签是 loss
    plt.plot(res.history['loss'], label='loss')
    
    # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
    plt.plot(res.history['val_loss'], label='val_loss')
    
    # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
    plt.legend(loc='upper right')
    
    # 展示图片
    plt.show()

    # 绘制模型训练过程中的的准确率和平均准确率
    # 绘制模型训练过程中的准确率曲线，标签是 acc
    plt.plot(res.history['accuracy'], label='accuracy')
    
    # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
    plt.plot(res.history['val_accuracy'], label='val_accuracy')
    
    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()
    
    # 展示图片
    plt.show()

def evaluate_mode(test_data, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_data: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------

    # ---------------------------------------------------------------------------


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
        # 开始时间
    start = time.time()

    # 数据预处理
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"

    save_model_path = 'results/net.h5'  # 保存模型路径和名称

    # 获取数据
    train_data, test_data = processing_data(data_path,227,227)

    # 创建、训练和保存模型
    res,model = train_model(train_data, test_data, save_model_path)
    load_and_model_prediction(test_data)
    #plot_training_history(res)
    # 评估模型
    #evaluate_mode(test_data, save_model_path)



if __name__ == '__main__':
    main()