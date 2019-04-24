import tensorflow as tf
from scipy.misc import imread, imresize
import VGG16_model as model

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
sess = tf.Session()
vgg = model.vgg16(imgs)
fc3_cat_and_dog = vgg.probs
pred = tf.nn.softmax(fc3_cat_and_dog)
saver = vgg.saver()
saver.restore(sess, 'model/')  # 加载保存的模型参数

import os

# 读取猫或者狗的测试图像
for root, sub_folders, files in os.walk('dog_cat_classification/test1'):
    i = 0
    cat = 0
    dog = 0
    result = []
    for name in files:
        i += 1
        filepath = os.path.join(root, name)

        try:
            img1 = imread(filepath, mode='RGB')
            img1 = imresize(img1, (224, 224))
        except:
            print("remove", filepath)

        prob = sess.run(pred, feed_dict={vgg.imgs: [img1]})

        import numpy as np

        max_index = np.argmax(prob)
        if max_index == 0:
            cat += 1
            result.append(0)
        else:
            dog += 1
            result.append(1)
        if (result[i-1] == 0):
            print("这是一只猫")
        else:
            print("这是一只狗")
        # 每50张图计算一次准确率
        if i % 50 == 0:
            acc = (cat * 1.) / (dog + cat)
            print(acc)
            print("-----------img number is %d------------" % i)
