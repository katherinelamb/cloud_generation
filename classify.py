import os
import argparse
from datetime import datetime
import pytz
import tensorflow as tf
import pdb
import numpy as np
import pandas as pd
from classifier_model import classifier
from hparam_manager import load_hparam_space, parse_hparams

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def create_dataset(files: list, labels: list, batch_size: int, training=True) -> (tf.data.Dataset, int):
    num_examples = len(labels)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(num_examples, seed=0)
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset, num_examples

def split_train_val(df, train_split, num_imgs):
    train_idxs = df.index.tolist()
    val_idxs = df.index.tolist()

    num_train = int(train_split * num_imgs)
    train_idxs = np.random.choice(train_idxs, num_train, replace=False)
    val_idxs = np.random.choice(val_idxs, num_imgs - num_train, replace=False)

    return train_idxs, val_idxs

def train(model: tf.keras.Model, data_csv: str, data_root: str, hparams: dict, hp_space: dict):
    train_split = hparams[hp_space['train_split']]
    num_imgs = hparams[hp_space['num_imgs']]
    num_epochs = hparams[hp_space['num_epochs']]
    batch_size = hparams[hp_space['batch_size']]

    # Load csv data
    examples_df = pd.read_csv(data_csv, header=0, skipinitialspace=True)

    label_dict = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}
    filenames = np.array([
        data_root + fname for fname in examples_df['Image'].tolist()
    ], dtype=object)
    labels = np.array([
        label_dict[fname] for fname in examples_df['Label'].tolist()
    ], dtype=int)

    if num_imgs == -1: num_imgs = len(filenames)

    train_idxs, val_idxs = split_train_val(examples_df, train_split, num_imgs)
    train_filenames = filenames[train_idxs]
    train_labels = labels[train_idxs]
    val_filenames = filenames[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset, num_train = create_dataset(
        files=train_filenames,
        labels=train_labels,
        batch_size=batch_size,
        training=True
    )

    val_dataset, num_val = create_dataset(
        files=val_filenames,
        labels=val_labels,
        batch_size=batch_size,
        training=False
    )

    print("Num train: {}, num val: {}".format(num_train, num_val))

    # log_dir = './logs/{}'.format(run_name)

    train_history = model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=num_train // batch_size,
        validation_steps=num_val // batch_size,
        validation_data=val_dataset,
        # callbacks=[
        #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        #     tf.keras.callbacks.TensorBoard(
        #         log_dir=log_dir),
        #     tf.keras.callbacks.ModelCheckpoint(
        #         filepath="experiments/{}/".format(run_name) +
        #         "best_model.hdf5",
        #         save_best_only=True,
        #         save_weights_only=False),
        #     hp.KerasCallback(log_dir, hparams)]
        )

def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    np.set_printoptions(precision=3)

    print("Num GPUs Available: ",
          len(tf.config.experimental.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--config_yaml',
                        type=str,
                        default='',
                        help='Path to experiment config file.')
    parser.add_argument('--csv_file',
                        type=str,
                        default='',
                        help='Path to csv file.')
    parser.add_argument('--data_root',
                        type=str,
                        default='',
                        help='Dataset root folder.')

    args = parser.parse_args()

    config_yaml = args.__dict__['config_yaml']
    csv_file = args.__dict__['csv_file']
    data_root = args.__dict__['data_root']

    hp_space = load_hparam_space()
    hparams = parse_hparams(config_yaml, hp_space)

    crop_size = hparams[hp_space['crop_size']]
    learning_rate = hparams[hp_space['learning_rate']]
    regularize = hparams[hp_space['regularize']]

    # run_name = "{}_{}_{}".format(
    #     model_name,
    #     config_yaml[config_yaml.rfind('/') + 1:config_yaml.find('.yaml')],
    #     datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d_%H%M%S"))

    model = classifier(input_size=(crop_size, crop_size, 3))

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_accuracy'])

    # experiment_dir = "experiments/{}/".format(run_name)
    # if not os.path.exists(experiment_dir):
    #     os.mkdir(experiment_dir)

    # Data config
    if csv_file == '' or data_root == '':
        raise FileNotFoundError(
            "Need to specify csv filepath and data root folder!")

    train(model, csv_file, data_root, hparams, hp_space)

if __name__ == '__main__':
    main()
