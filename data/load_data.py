import tensorflow as tf
import numpy as np
from matplotlib import pylab as plt
import os


def _parse_record(example_photo):
    """
    :param example_photo:  是序列化后的数据
    :return: 反序列化的数据
    """
    # 定义一个解析序列的features
    expected_features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)
    }
    # 反序列化
    parsed_features = tf.parse_single_example(example_photo, features=expected_features)
    # 将数据图片从字符串解析还原
    shape = [30, 512, 512, 1]
    # feature = {
    #     "image": tf.reshape(tf.decode_raw(parsed_features["image"], tf.float16), shape=shape),
    #     "mask": tf.reshape(tf.decode_raw(parsed_features["mask"], tf.float16), shape=shape),
    #     "shape": parsed_features["shape"]
    # }
    # return feature
    img = tf.cast(tf.reshape(tf.decode_raw(parsed_features["image"], tf.float), shape=shape), dtype=tf.float32)
    musk = tf.cast(tf.reshape(tf.decode_raw(parsed_features["mask"], tf.float16), shape=shape), dtype=tf.float32)
    return tf.image.resize_images(img, size=[256, 256]), tf.image.resize_images(musk, size=[256, 256])


def _parse_record2(example_photo):
    """
    :param example_photo:  是序列化后的数据
    :return: 反序列化的数据
    """
    # 定义一个解析序列的features
    expected_features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "shape": tf.io.FixedLenFeature(shape=[4], dtype=tf.int64)
    }
    # 反序列化
    parsed_features = tf.parse_single_example(example_photo, features=expected_features)
    # 将数据图片从字符串解析还原
    # shape = [20, 512, 512, 1]
    shape = parsed_features["shape"]
    img = tf.cast(tf.reshape(tf.decode_raw(parsed_features["image"], tf.float64), shape=shape), dtype=tf.float32)
    musk = tf.cast(tf.reshape(tf.decode_raw(parsed_features["mask"], tf.uint16), shape=shape), dtype=tf.float32)

    # 对img进行变换
    img = tf.pow(tf.add(tf.divide(img, 2), 0.5), 0.25)

    return tf.image.resize_images(img, size=[512, 512]), tf.image.resize_images(musk, size=[512, 512])


def tfrecord_to_dataset(file_names, BATCH_SIZE=1):
    """
    :param file_names: filenames list:train or test
    :param batch_size: batch_size
    :return: dataset
    """
    # 2.构建文件dataset
    dataset_file = tf.data.Dataset.list_files(file_names).repeat()

    # 3.构建全部文件内容的dataset_filecontent
    dataset_filecontent = dataset_file.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=3  # 读取文件的并行数
    )
    # dataset_filecontent = dataset_filecontent.shuffle(10)   # 打乱文件

    # 3.构建样本的dataset
    dataset = dataset_filecontent.map(_parse_record2,  # 负责将example解析并反序列化处理的函数
                                      num_parallel_calls=3  # 处理样本的并行线程数量
                                      )
    dataset = dataset.repeat().batch(BATCH_SIZE)
    return dataset


def data_generator(tfrecord_dir, flag="train"):
    filenames = os.listdir(tfrecord_dir)
    filenames = [os.path.join(tfrecord_dir, filename) for filename in filenames]

    if flag == "train":
        # 训练的时候一次产生20个样本
        dataset = tfrecord_to_dataset(filenames, BATCH_SIZE=5)
    else:
        # 测试的时候一次产生20*2个样本
        dataset = tfrecord_to_dataset(filenames, BATCH_SIZE=2)

    iterator = dataset.make_one_shot_iterator()
    images_, masks_ = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        images, masks = sess.run([images_, masks_])
        while True:
            images = images.reshape([-1, 512, 512, 1])
            masks = masks.reshape([-1, 512, 512, 1])

            # random
            permutation = np.random.permutation(images.shape[0])
            images = images[permutation]
            masks = masks[permutation]

            # if flag == "train":
            #     for i in range(10):
            #         yield images[i*20:(i+1)*20], masks[i*20:(i+1)*20]
            # else:
            #     yield images, masks

            if flag == "train":
                for i in range(20):
                    yield images[i*5:(i+1)*5], masks[i*5:(i+1)*5]
            else:
                for i in range(8):
                    yield images[i*10:(i+1)*10], masks[i*10:(i+1)*10]


def get_dataset(tfrecord_dir, BATCH_SIZE):
    # 1.获取tfrecord文件名
    filenames = os.listdir(tfrecord_dir)
    train_filenames = [os.path.join(tfrecord_dir, filename) for filename in filenames if filename.startswith("train")]
    test_filenames = [os.path.join(tfrecord_dir, filename) for filename in filenames if filename.startswith("test")]

    train_dataset = tfrecord_to_dataset(train_filenames, BATCH_SIZE)
    test_dataset = tfrecord_to_dataset(test_filenames, 100)
    return train_dataset, test_dataset


def main():
    # 1.获取tfrecord文件名
    train_tfrecord_dir = "/home/sxd/wangmeng/data/COVID-19-CT-Seg_20cases/tfrecord2/gaussian"
    test_tfrecord_dir = "/home/sxd/wangmeng/data/COVID-19-CT-Seg_20cases/tfrecord2/test"

    train_filenames = os.listdir(train_tfrecord_dir)
    test_filenames = os.listdir(test_tfrecord_dir)

    train_filenames = [os.path.join(train_tfrecord_dir, filename) for filename in train_filenames]
    test_filenames = [os.path.join(test_tfrecord_dir, filename) for filename in test_filenames]

    train_dataset = tfrecord_to_dataset(train_filenames, 6)
    test_dataset = tfrecord_to_dataset(test_filenames, 6)
    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    batch = 0
    with tf.Session() as sess:
        images, masks, = sess.run(train_iterator.get_next())
        if not os.path.exists("images"):
            os.mkdir("images")
        for step in range(5):
            plt.imshow(images[0, step, :, :, 0])
            plt.savefig("images/{}-image.png".format(step))
            plt.clf()
            plt.imshow(masks[0, step, :, :, 0])
            plt.savefig("images/{}-mask.png".format(step))
            plt.clf()
    return None


if __name__ == "__main__":
    main()


