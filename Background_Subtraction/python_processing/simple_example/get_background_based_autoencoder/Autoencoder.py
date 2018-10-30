"""
Implementation of Convolutional Autoencoder using Keras

Created on 24.08.2018

@author: klickmal
"""

"""
corrected on 30.10.2018

@corrector: qing
"""

import keras 
from keras.models import Model, load_model
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers.pooling import MaxPool2D
from data import Data
import numpy as np
from skimage import io

class convolutional_Autoencoder():
    """
    Provide the structure of autoencoder using keras
    """
    def __init__(self, image_shape):
        self.image_shape = image_shape
    
    def autoencoder(self, x):
        x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'conv1', data_format = 'channels_last')(x)
                
        x = MaxPool2D((2,2), strides = (2,2), name = 'max1')(x)

        x = Conv2D(32, (3, 3), activation = 'relu',  padding = 'same', name = 'conv2')(x)
        x = MaxPool2D((2, 2), strides = (2,2), name = 'max2')(x)

        x = Flatten()(x)
        x = Dense(20)(x)  # output 20 dimensional vector

 #       x = Dense(75*120*32)(x)
 #       x = Reshape((75, 120, 32))(x)
        x = Dense((self.image_shape[0]/4)*(self.image_shape[1]/4)*32)(x)
        x = Reshape((self.image_shape[0]/4, self.image_shape[1]/4, 32))(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, (3, 3), activation = 'relu', padding = 'same', name = 'conv_t11')(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(1, (3, 3), activation = 'sigmoid', padding = 'same', name = 'conv_t12')(x)

        return x
    
    def init_Model(self):
        h, w, d = self.image_shape
        input_image = Input(shape=(h, w, d), name = 'input')

        output_image = self.autoencoder(input_image)

        model = Model(inputs = input_image, outputs = output_image, name = 'ae')
        
        opt = keras.optimizers.RMSprop(lr = 1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model
    
if __name__ == '__main__':

    def train(train_img_path, saved_path):

        data = Data()
        train_images,[row,col,img_num] = data.get_image(train_img_path)
        train_images = train_images.reshape((img_num, row, col, 1))

        _, h, w, d = train_images.shape
        image_shape = (h, w, d)

        model = convolutional_Autoencoder(image_shape)
        model = model.init_Model()

        epoch = 2
        batch_size = 1
        
        chk = keras.callbacks.ModelCheckpoint(saved_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        model.fit(train_images, train_images,  validation_split=0.2, batch_size = batch_size, epochs = epoch, callbacks = [chk])
  
    def test(test_img_path, saved_path, saved_image):

        data = Data()
        test_images,[row,col,img_num] = data.get_image(test_img_path)
        test_images = test_images[:, :, :, np.newaxis]
        test_images = test_images.reshape((img_num, row, col, 1))

        model = load_model(saved_path)
        predicted_image = model.predict(test_images, batch_size = 1)
        print('predict: {predicted_image.shape}')
        for index, image in enumerate(predicted_image):
            saved_images = saved_image + '/' + str(index) + '.jpg'
            image = np.squeeze(image, axis = 2)
            io.imsave(saved_images, image)

    saved_path = './model_save/autoencoder.h5'
    train_img_path = './train_image'
    test_img_path = './test_image'
    saved_img = './save_image'

    train(train_img_path, saved_path)
    test(test_img_path, saved_path, saved_img)

