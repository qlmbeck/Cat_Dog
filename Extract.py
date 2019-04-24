import os
import shutil

# 下载得到的训练集图像
image_path = './train'
# 将猫狗分类保存的路径
train_path = './dog_cat_classification'

image_list = os.listdir(image_path)
# 读取1000张猫狗图像，按照图像名字分别保存
for image_name in image_list[-1000:-1]:
    class_name = image_name[0:3]
    save_path = os.path.join(train_path, class_name)
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)
    file_name = os.path.join(image_path, image_name)
    save_name = os.path.join(save_path, image_name)
    shutil.copyfile(file_name, save_name)
