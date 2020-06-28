from data import data_generator
import os
import matplotlib.pyplot as plt
import keras
from utils import weighted_dice_coefficient_loss, dice_coefficient_loss
import numpy as np

GPU_SET = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET  # 设置当前程序仅能看到1个gpu
train_tfrecord_dir = "/home/wangmeng/Data/COVID-19-CT-Seg_20cases/raw"
test_tfrecord_dir = "/home/wangmeng/Data/COVID-19-CT-Seg_20cases/test"


def predict(model_name):
    generator_train = data_generator(train_tfrecord_dir, "test")
    model = keras.models.load_model(model_name)
                                    # custom_objects={"weighted_dice_coefficient_loss": weighted_dice_coefficient_loss})
    model.summary()
    images, masks = generator_train.__next__()

    pred = model.predict(images[:5])

    if not os.path.exists("predict"):
        os.mkdir("predict")

    for i in range(5):
        plt.imshow(pred[i, :, :, 0], cmap="gray")
        plt.savefig("predict/pred"+str(i)+'.png')
        plt.clf()

        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.savefig("./predict/image"+str(i)+".png")
        plt.clf()

        plt.imshow(masks[i, :, :, 0], cmap="gray")
        plt.savefig("./predict/mask"+str(i)+".png")
        plt.clf()

        mask = np.greater(pred[i, :, :, 0], 0.5).astype('float')
        plt.imshow(mask, cmap="gray")
        plt.savefig("./predict/mask_reslut"+str(i)+".png")
        plt.clf()
    return None


if __name__ == '__main__':
    model_name = "ckpt/004-0.0435.h5"
    predict(model_name)
