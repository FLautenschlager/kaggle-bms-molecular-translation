import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt


def create_dataset(path, image_size=(180, 180), batch_size=10):
    print("Starting to load image dataset")
    path = path + '**/*.png'
    filelist = glob(path, recursive=True)
    image_count = len(filelist)
    print("\tImage count: {}".format(image_count))
    list_ds = tf.data.Dataset.from_tensor_slices(filelist)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(5):
        print(f.numpy())

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())

    autotune = tf.data.AUTOTUNE

    def process_path(file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img, imgsize=image_size)
        return img, img

    train_ds = train_ds.map(process_path, num_parallel_calls=autotune)
    val_ds = val_ds.map(process_path, num_parallel_calls=autotune)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)

    train_ds = configure_for_performance(train_ds, batch_size=batch_size, autotune=autotune)
    val_ds = configure_for_performance(val_ds, batch_size=batch_size, autotune=autotune)

    return train_ds, val_ds


def decode_img(img, imgsize=(180, 180)):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # resize the image to the desired size
    return tf.image.resize(img, [imgsize[0], imgsize[1]])


def configure_for_performance(ds, batch_size=10, autotune=tf.data.AUTOTUNE):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=autotune)
    return ds


if __name__ == "__main__":
    train, val = create_dataset("../data/bms-molecular-translation/train/")

    image_batch, label_batch = next(iter(train))

    plt.figure(figsize=(10, 10))

    print("\tplot images.")
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()
