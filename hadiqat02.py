#!/usr/bin/env python3

# Standard Packages
import os

# Data Science Packages
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Packages
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

# Model Plotting
from keras.utils.vis_utils import plot_model

# Dataset
from keras.datasets.mnist import load_data

def configure_generator(latent_shape):
    model = Sequential()
    # Input Layer / 25% nodes downsample
    num_nodes = 128 * 7 * 7
    model.add(Dense(num_nodes, input_dim=latent_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # Hidden Layer 1 / 50% nodes upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Hidden Layer 2 / 100% nodes upsample
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Output Layer / ? node / sigmoid activation
    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

    return model

def configure_discriminator(input_shape=(28, 28, 1)):
    model = Sequential()
    # Hidden Layer 1 / 64 nodes / 3 * 3 * 1 kernel / 2 * 2 strides
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    # Hidden Layer 2 / 64 nodes / 3 * 3 * 1 kernel / 2 * 2 strides
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    # Preprocessing
    model.add(Flatten())
    # Output Layer / 1 node / sigmoid activation
    model.add(Dense(1, activation='sigmoid'))
    # Compile Neural Network
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    return model

def configure_generative_adversarial_network(generator, discriminator):
    discriminator.trainable = False #Turn off training of weights in discriminator
    model = Sequential()
    # Generator
    model.add(generator)
    # Discriminator
    model.add(discriminator)
    # Compile Generative Adversarial Network
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return model

def load_real_samples():
    (train_set, _), (_, _) = load_data()
    X = expand_dims(train_set, axis=-1)
    X = X.astype('float32')
    X = X / 255.0

    return X

def generate_latent_points(latent_shape, num_samples):
    X_input = np.randn(latent_shape * num_samples)
    X_input = X_input.reshape(num_samples, latent_shape)

    return X_input

def generate_real_samples(dataset, num_samples):
    x = np.randint(0, dataset.shape[0], num_samples)
    X = dataset[x]
    y = np.ones((num_samples, 1))

    return X, y

def generate_false_samples(generator, latent_shape, num_samples):
    X_input = generate_latent_points(latent_shape, num_samples)
    X = generator.predict(X_input)
    y = np.zeros((num_samples, 1))

    return X, y

def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1, i + 1)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = './generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()

def summarise_performance(epoch, generator, discriminator, dataset, latent_shape, num_samples=100):
    # Prepare real samples
    X_real, y_real = generate_real_samples(dataset, num_samples)
    # Evaluate discriminator on real samples
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    # Prepare false samples
    X_false, y_false = generate_false_samples(generator, latent_shape, num_samples)
    # Evaluate discriminator on false samples
    _, acc_false = discriminator.evaluate(X_false, y_false, verbose=0)
    # Summarise discriminator performance
    print('>Accuracy [real]: %.0f%%, [false]: %.0f%%' % (acc_real * 100, acc_false * 100))
    save_plot(X_false, epoch)
    filename = './generator_model_%03d.h5' % (epoch + 1)
    generator.save(filename)

def train(generator, discriminator, generative_adversarial_network, dataset, latent_shape, num_epochs=100, num_batch=256):
    batch_per_epoch = int(dataset.shape[0] / num_batch)
    half_batch = int(num_batch / 2)
    for i in range(num_epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_false, y_false = generate_false_samples(generator, latent_shape, half_batch)
            X, y = np.vstack((X_real, X_false)), np.vstack((y_real, y_false))
            discriminator_loss, _ = discriminator.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_shape, num_batch)
            y_gan = np.ones((num_batch, 1))
            generator_loss = generative_adversarial_network.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, batch_per_epoch, discriminator_loss, generator_loss))
        if (i + 1) % 10 == 0:
            summarise_performance(i, generator, discriminator, dataset, latent_shape)

latent_shape = 100

# Create Discriminator
discriminator = configure_discriminator()

# Create Generator
generator = configure_generator(latent_shape)

# Create Generative Adversarial Network
hadiqat02 = configure_generative_adversarial_network(generator, discriminator)

# Summarise Generative Adversarial Network
hadiqat02.summary()

# Plot Generative Adversarial Network
plot_model(hadiqat02, to_file='./hadiqat02_plot.png', show_shapes=True, show_layer_names=True)

# Load Image Data
dataset = load_real_samples()

# Train Model
train(generator, discriminator, hadiqat02, dataset, latent_shape)
