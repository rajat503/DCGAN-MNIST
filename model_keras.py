from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.layers.core import Reshape
import numpy as np

def get_discriminator():
    D = Sequential()
    D.add(Convolution2D(32, 3, 3, border_mode='valid',subsample=(2,2), input_shape=(28,28,1)))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2)))
    # D.add(BatchNormalization(epsilon=0.001, mode=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(2,2)))
    # D.add(BatchNormalization(epsilon=0.001, mode=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    return D

def get_generator():
    G = Sequential()
    G.add(Dense(input_dim=100, output_dim=128*7*7))
    G.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    G.add(BatchNormalization(epsilon=0.001, mode=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    G.add(UpSampling2D(size=(2, 2)))
    G.add(Convolution2D(64, 3, 3, border_mode='same'))
    G.add(BatchNormalization(epsilon=0.001, mode=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
    G.add(Activation('relu'))
    G.add(UpSampling2D(size=(2, 2)))
    G.add(Convolution2D(1, 3, 3, border_mode='same'))
    G.add(Activation('tanh'))
    return G

def get_dcgan(generator, discriminator):
    dcgan = Sequential()
    dcgan.add(generator)
    discriminator.trainable = False
    dcgan.add(discriminator)
    return dcgan

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

g_opt = Adam(lr=0.0002, beta_1=0.5)
d_opt = Adam(lr=0.0002, beta_1=0.5)
# d_opt = SGD(lr=0.001, momentum=0.9, nesterov=True)

generator = get_generator()
discriminator = get_discriminator()
dcgan = get_dcgan(generator, discriminator)
generator.compile(loss='binary_crossentropy', optimizer=g_opt)
dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)


BATCH_SIZE = 128
# disk_list =[]

noise = np.zeros((BATCH_SIZE, 100))

for epoch in range(100):
    for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.normal(0, 0.5, 100)
        image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        generated_images = generator.predict(noise)
        # X = np.concatenate((image_batch, generated_images))
        # y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        if index%2==0:
            X = image_batch
            y = [1] * BATCH_SIZE
        else:
            X = generated_images
            y = [0] * BATCH_SIZE
        d_loss = discriminator.train_on_batch(X, y)
        print("epoch %d batch %d d_loss : %f" % (epoch, index, d_loss))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.normal(0, 0.5, 100)
        discriminator.trainable = False
        g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)
        discriminator.trainable = True
        print("epoch %d batch %d g_loss : %f" % (epoch, index, g_loss))

for i in range(BATCH_SIZE):
    noise[i, :] = np.random.normal(0, 0.5, 100)
generated_images = generator.predict(noise)
    # for i in generated_images:
        # disk_list.append(generated_image)

import cv2
for i in generated_images:
    img = cv2.imshow("a",i)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
