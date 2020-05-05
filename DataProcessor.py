from keras.preprocessing.image import ImageDataGenerator

# 定义数据预处理器类
class DataProcessor(object):
    def __init__(self,data_path, target_size, batch_size=32, validation_split=0.1):
        # 带有子目录的数据集路径
        self.data_path = data_path
        # 目标图像大小,(height,width)
        self.target_size = target_size
        # batch 数据的大小，整数，默认32。
        self.batch_size = batch_size
        # 在 0 和 1 之间浮动。用作测试集的训练数据的比例，默认0.1。
        self.validation_split = validation_split
    def get_train_generator(self):
        train_data = ImageDataGenerator(
            # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
            rescale=1. / 255,
            # 浮点数，剪切强度（逆时针方向的剪切变换角度）
            shear_range=0.1,  
            # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
            zoom_range=0.1,
            # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
            width_shift_range=0.1,
            # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
            height_shift_range=0.1,
            # 布尔值，进行随机水平翻转
            horizontal_flip=True,
            # 布尔值，进行随机竖直翻转
            vertical_flip=True,
            # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
            validation_split=self.validation_split  
        )

        train_generator = train_data.flow_from_directory(
            # 提供的路径下面需要有子目录
            data_path, 
            # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
            target_size=target_size,
            # 一批数据的大小
            batch_size=batch_size,
            # "categorical", "binary", "sparse", "input" 或 None 之一。
            # 默认："categorical",返回one-hot 编码标签。
            class_mode='categorical',
            # 数据子集 ("training" 或 "validation")
            subset='training', 
            seed=0)
        return train_generator
    def get_validation_generator(self):
        validation_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=self.validation_split)

        validation_generator = validation_data.flow_from_directory(
            self.data_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=0)

    return train_generator, validation_generator