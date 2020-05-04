from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os


# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
model_path = None

# 加载模型，如果采用keras框架训练模型，则 model=load_model(model_path)
model = None
    
# ---------------------------------------------------------------------------

def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理
        2.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别, 
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 获取图片的类别，共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    # 把图片转换成为numpy数组
    img = image.img_to_array(img)
    

    # 获取输入图片的类别
    y_predict = None

    # -------------------------------------------------------------------------
    
    # 返回图片的类别
    return y_predict

from keras.preprocessing import image

# 输入图片路径和名称
img_path = 'test.jpg'

# 打印该张图片的类别
img = image.load_img(img_path)
print(predict(img))