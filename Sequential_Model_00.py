# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:57:43 2019

@author: Wilson
reference: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras

#init
#use gpu 3
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '3'

#######################################
#       Data Preparation
#######################################

    ###---  dimensions of our images.
img_width, img_height = 256, 256

#train_data_dir = 'C:/Users/Wilson/Documents/Seunghye/02_AgileBeat_models/TRAIN'
#validation_data_dir = 'C:/Users/Wilson/Documents/Seunghye/02_AgileBeat_models/TEST'

train_data_dir = '/playpen/wilson/Seung_images/TRAIN'
validation_data_dir = '/playpen/wilson/Seung_images/TEST'

nb_train_samples = 2036
nb_validation_samples = 1113
epochs = 50
batch_size = 16


    ###---  configure the shape of dataset
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

#######################################
#       Model
#######################################
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop',
             # optimizer='adam',
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height), 
                                                    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("sequential.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


    ###--- Model fitting and Validation
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks = [checkpoint, early])











