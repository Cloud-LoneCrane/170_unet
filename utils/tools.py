import os
import matplotlib.pyplot as plt
global counter


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


def draw_net(model, filename):
    from keras.utils import plot_model
    plot_model(model, filename)

    return None