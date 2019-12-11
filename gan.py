# https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

from __future__ import print_function, division

# from keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from PIL import Image

import matplotlib.pyplot as plt

import sys
import os
import pandas as pd

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        BASE_DIR = os.path.dirname((os.path.dirname(__file__)))
        self.train_csv = os.path.join(BASE_DIR, 'inputs/crops/train_crops.csv')
        self.train_dir = os.path.join(BASE_DIR, 'inputs/crops/train_crops/')
        self.n_train = 4000
        self.label = 'Gravel'
        self.synthetic_csv = os.path.join(BASE_DIR, 'inputs/crops/synthetic_crops_{}.csv'.format(self.label))

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = np.empty((self.n_train, self.img_shape[0], self.img_shape[1], self.channels))
        train_df = pd.read_csv(self.train_csv, skipinitialspace=True)
        cnt = 0
        for i in range(len(train_df)):
            if cnt == self.n_train: break
            label = train_df.iloc[i]['Label']
            if label != self.label: continue
            img_path = os.path.join(self.train_dir, train_df.iloc[i]['Image'])
            if not os.path.exists(img_path): continue
            img = Image.open(img_path)
            arr = np.array(img)
            X_train[cnt] = arr
            cnt += 1

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # self.plot_samples(imgs)
            # exit()

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                save_model(self.combined, 'models/gan_{}_{}.h5'.format(self.label, epoch), save_format='h5')

            if (epoch + 1) == epochs:
                self.save_samples()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/{}_{}.png".format(self.label, epoch))
        plt.close()

    def save_samples(self):
        num_samples = 1000
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = (255 * gen_imgs).astype(np.uint8)
        outdir = 'inputs/crops/synthetic_crops/'
        os.makedirs(outdir, exist_ok=True)

        with open(self.synthetic_csv, 'a') as file:
            for i in range(num_samples):
                filename = '{}_{}.jpg'.format(self.label, i)
                file.write('{},{}\n'.format(filename, self.label))
                out_path = os.path.join(outdir, filename)
                Image.fromarray(gen_imgs[i]).save(out_path)

    def plot_samples(self, imgs):
        r, c = 5, 5
        # Rescale images 0 - 1
        gen_imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        plt.show()

if __name__ == '__main__':
    os.makedirs('images/', exist_ok=True)
    os.makedirs('models/', exist_ok=True)
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
