from data import data_generator
from model import uent2d_model, get_callbacks, uent2d_dilate_model, uent2d_dilate_model_1X1
import os
import matplotlib.pyplot as plt
import keras
from utils import weighted_dice_coefficient_loss, dice_coefficient_loss

GPU_SET = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_SET  # 设置当前程序仅能看到1个gpu
train_tfrecord_dir = "/home/wangmeng/Data/COVID-19-CT-Seg_20cases/raw"
test_tfrecord_dir = "/home/wangmeng/Data/COVID-19-CT-Seg_20cases/test"

epochs = 300
total_serial = 498
total_example = total_serial*20
batch = 10
steps_per_epoch = total_example/batch

counter = 0


def save_img(images, masks, num):
    if not os.path.exists("./images"):
        os.mkdir("./images")
    for i in range(num):
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.savefig("./images/image"+str(i)+"-"+str(counter)+".png")
        plt.clf()

        plt.imshow(masks[i, :, :, 0], cmap="gray")
        plt.savefig("./images/mask"+str(i)+"-"+str(counter)+".png")
        plt.clf()
    return None


def train():
    generator_train = data_generator(train_tfrecord_dir, "train")
    generator_test = data_generator(test_tfrecord_dir, "test")
    images, masks = generator_train.__next__()
    # save_img(images, masks)

    input_shape = (512, 512, 1)
    model = uent2d_model(input_shape=input_shape, loss_function=keras.losses.binary_crossentropy)
    # 打印模型结构
    model.summary()

    # # 保存模型图
    # from keras.utils import plot_model
    # plot_model(model, to_file="model.png")

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs',  # log 目录
                                              histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                              batch_size=5,      # 用多大量的数据计算直方图
                                              write_graph=True,  # 是否存储网络结构图
                                              write_grads=True,  # 是否可视化梯度直方图
                                              write_images=True,  # 是否可视化参数
                                              embeddings_freq=0,
                                              embeddings_layer_names=None,
                                              embeddings_metadata=None)
    if not os.path.exists("./ckpt"):
        os.mkdir("./ckpt")
    filepath = "./ckpt/{epoch:03d}-{val_loss:.4f}.h5"
    callbacks = get_callbacks(filepath)
    callbacks.append(tensorboard)
    model.fit_generator(generator_train, validation_data=generator_test, steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks, epochs=epochs, verbose=1, validation_steps=1)
    model.save("./ckpt/save_model.h5")
    return None


def predict(model_name):
    generator_train = data_generator(train_tfrecord_dir, "train")
    model = keras.models.load_model(model_name,
                                    custom_objects={"weighted_dice_coefficient_loss": weighted_dice_coefficient_loss})
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

    return None


def draw_net(model, filename):
    from keras.utils import plot_model
    plot_model(model, filename)

    return None


if __name__ == '__main__':
    # train()
    # predict("ckpt/001-0.0104.h5")
    pass
