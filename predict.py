from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

model_path = "results/net.h5"
model = load_model(model_path)

def predict(img):
    model_path = 'results/net.h5'
    model = load_model(model_path)
    img = image.img_to_array(img)
    img = 1.0/255 * img
    x = np.expand_dims(img, axis=0)
    # 获取输入图片的类别
    y_predict = None
    # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
    x = np.expand_dims(img, axis=0)

    # 模型预测
    y = model.predict(x)

    # 获取labels
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

    # 获取输入图片的类别
    y_predict = labels[np.argmax(y)]

    # 返回图片的类别
    return y_predict

# 输入图片路径和名称
img_path = 'datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized/cardboard/cardboard2.jpg'
# 打印该张图片的类别
img = image.load_img(img_path,target_size=(227,227))

print(predict(img))