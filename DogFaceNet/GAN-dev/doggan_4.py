from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import sys

import numpy as np

import os
import skimage as sk
from tqdm import tqdm


PATH = '../data/dogfacenet/aligned/after_4_resized/'
PATH_SAVE = '../output/images/dcgan/dogs/'
PATH_MODEL = '../output/model/gan/'
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 127
        self.img_cols = 127
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer_d = Adam(0.0001, 0.2)
        optimizer_c = Adam(0.01, 0.2)
        # optimizer = RMSprop(lr=0.005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_c)

    def build_generator(self):

        model = Sequential()

        model.add(Reshape((1,1,self.latent_dim)))

        # model.add(Conv2DTranspose(512,(6,6)))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))

        # filters = [256,128,64,32]
        # kernels = [3,3,3,3]

        # for i in range(len(filters)):
        #     model.add(Conv2DTranspose(filters[i],kernels[i],strides=(2,2)))
        #     model.add(BatchNormalization(momentum=0.8))
        #     model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2DTranspose(3,(4,4),strides=(2,2)))
        # model.add(Activation('tanh'))

        model.add(Conv2DTranspose(256,(3,3)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        filters = [128,64,32,16]
        kernels = [3,3,3,3]

        for i in range(len(filters)):
            model.add(Conv2DTranspose(filters[i],kernels[i],strides=(2,2)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(3,(3,3),strides=(2,2)))
        model.add(Activation('tanh'))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        model.summary()

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))


        # for layer in [32,64,64,64,256]:
        #     model.add(Conv2D(layer, kernel_size=3, strides=2))
        #     model.add(BatchNormalization(momentum=0.8))
        #     model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2D(512, kernel_size=2, strides=1))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))


        for layer in [32,64,128,128]:
            model.add(Conv2D(layer, kernel_size=3, strides=2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=3, strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)


        # Load datas
        print("Load data into memory...")
        filenames = np.empty(0)
        idx = 0
        for root,_,files in os.walk(PATH):
            if len(files)>1:
                for i in range(len(files)):
                    files[i] = root + '/' + files[i]
                filenames = np.append(filenames,files)

        # max_size = len(filenames)
        max_size = len(filenames)
        X_train = np.empty((max_size,self.img_cols,self.img_rows,self.channels))
        for i,f in tqdm(enumerate(filenames)):
            if i == max_size:
                break
            X_train[i] = sk.io.imread(f)/ 127. - 1.
        print("done")


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.combined.save(PATH_MODEL+"2019.05.17.doggan_4."+str(epoch)+".h5")

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(PATH_SAVE+"mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=40000, batch_size=32, save_interval=200)
