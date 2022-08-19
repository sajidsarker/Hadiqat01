import os

import numpy as np
import matplotlib as plt

from keras.models import Sequential
#from keras.models import save_model
#from keras.models import save_weights
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.utils.vis_utils import plot_model

from PIL import Image
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import array_to_img

class GenerativeAdversarialNetwork():
    def __init__(self, path_to_input, input_mask, input_shape, latent_space):
        print('Input Shape: {}\n Latent Shape: {}\nTraining Image Data: {}/*{}'.format(input_shape, latent_space, path_to_input, input_mask))
        self.input_shape = input_shape
        self.latent_space = latent_space
        #self.data = self.load_data(path_to_input, input_mask)
        # Configure our component and composite models
        print('[1] Configuring Generator ...')
        self.generator = self.configure_generator()
        print('[2] Configuring Discriminator ...')
        self.discriminator = self.configure_discriminator()
        print('[3] Configuring Generative Adversarial Network ...')
        self.model = self.configure_generative_adversarial_network()

    def load_data(self, path_to_input, input_mask):
        files = os.listdir(path_to_input)
        #files = files[:25]
        # Applying input mask to list of files in directory
        for file in files:
            if file[-3] != input_mask:
                files.remove(file)
        # Declaring dataset
        data = np.zeros((len(files), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        i = 0
        # Loading dataset
        for file in files:
            data[i, :, :, :] = read(file)
            i += 1
        # Preprocessing (Normalisation) dataset
        data = data.astype('float32')
        data = data / 255.0
        return data
    
    def export(self, path_to_output):
        '''save_model(
            model=self.generator,
            filepath=path_to_output,
            overwrite=True,
            include_optimizer=True,
            save_format='tf',
            signatures=None,
            options=None
        )
        save_weights(
            model=self.generator,
            filepath=path_to_output,
            overwrite=True,
            save_format='tf',
            options=None
        )'''

    def configure_generator(self, num_variation=128):
        model = Sequential()
        # Input Layer / 25% nodes downsample
        num_nodes = num_variation * self.input_shape[0] * self.input_shape[1] * 0.25 * 0.25 * self.input_shape[2]
        num_nodes = int(num_nodes)
        model.add(Dense(num_nodes, input_dim=self.latent_space))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape(int(self.input_shape[0] * 0.25), int(self.input_shape[1] * 0.25), self.input_shape[2], num_variation))
        # Hidden Layer 1 / 50% nodes upsample
        model.add(Conv2DTranspose(num_variation, kernel_size=4, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # Hidden Layer 2 / 100% nodes upsample
        model.add(Conv2DTranspose(num_variation, kernel_size=4, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # Output Layer
        model.add(Conv2D(self.input_shape[2], kernel_size=(int(self.input_shape[0] * 0.25), int(self.input_shape[1] * 0.25)), activation='sigmoid', padding='same'))
        display(model.summary())
        return model

    def configure_discriminator(self):
        model = Sequential()
        # Hidden Layer 1 / 64 nodes / 3 * 3 * 3 kernel / 2 * 2 strides
        model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same', input_shape=self.input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        # Hidden Layer 2 / 64 nodes / 3 * 3 * 3 kernel / 2 * 2 strides
        model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        # Preprocessing
        model.add(Flatten())
        # Output Layer / 1 node / sigmoid activation
        model.add(Dense(1, activation='sigmoid'))
        # Compile Neural Network
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy', 'precision', 'recall'])
        display(model.summary())
        return model

    def configure_generative_adversarial_network(self):
        self.discriminator.trainable = False
        model = Sequential()
        # Generator
        model.add(self.generator)
        # Discriminator
        model.add(self.discriminator)
        # Compile Generative Adversarial Network
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        display(model.summary())
        return model
    
    def sample_latent_points(self, num_sample):
        data = randn(self.latent_space * num_sample)
        data = data.reshape(num_sample, self.latent_space)
        return data
        
    def sample_truth(self, num_sample=128):
        observations = randint(0, self.data.shape[0], num_sample)
        data = self.data[observations]
        target = ones((num_sample, 1))
        return data, target
    
    def sample_false(self, num_sample=128):
        data = self.sample_latent_points(num_sample)
        data = self.generator.predict(data)
        target = zeros((num_sample, 1))
        return data, target
    
    def train(self, num_epoch=100, num_batch=256):
        for i in range(num_epoch):
            for j in range(int(self.data.shape[0] / num_batch)):
                data_truth, target_truth = self.sample_truth(self.data, int(num_batch * 0.5))
                data_false, target_false = self.sample_false(self.data, int(num_batch * 0.5))
                data, target = vstack((data_truth, data_false)), vstack((target_truth, target_false))
                discriminator_loss, _ = self.discriminator.train_on_batch(data, target)

                data = self.sample_latent_points(num_batch)
                target = ones((num_batch, 1))
                generator_loss = self.model.train_on_batch(data, target)

                print("Epoch [{}]: Batch [{} / {}] / Discriminator Loss={}%, Generator Loss={}%".format(i+1, j+1, int(self.data.shape[0] / num_batch), discriminator_loss, generator_loss))

hadiqat01 = GenerativeAdversarialNetwork('~/Desktop/Hadiqat01/training-image-data', '.JPG', (500, 500, 3), 100)
#hadiqat01.train()
#hadiqat01.export('~/Desktop/Hadiqat01/model-weights')
