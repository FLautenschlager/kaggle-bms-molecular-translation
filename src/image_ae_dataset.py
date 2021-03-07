import tensorflow as tf
import pathlib


def create_dataset(path, image_size=(180, 180)):
    data_dir = pathlib.Path(path)
    image_count = len(list(data_dir.glob('*/*.png')))
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(5):
        print(f.numpy())

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())


if __name__ == "__main__":
    create_dataset("../data/bms-molecular-translation/train/")
