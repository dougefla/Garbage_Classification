from Model import Model
from DataProcessor import DataProcessor
import time

def main():

    # 数据路径
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"
    # 获取数据
    dataprocessor = DataProcessor(data_path, target_size=(227,227),batch_size=32,validation_split=0.1)
    # 建立训练集数据生成器
    train_data = dataprocessor.get_train_generator()
    # 建立验证集数据生成器
    validation_data = dataprocessor.get_validation_generator()

    # 建立模型
    model = Model(model_type='alexnet',label_num=6)
    # 训练模型
    model.train(train_data,validation_data,epochs=300)
    # 打印训练过程
    model.plot_training_history()

if __name__ == '__main__':
    main()