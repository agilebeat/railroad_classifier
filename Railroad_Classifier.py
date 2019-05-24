# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:33:22 2019

@author: Wilson
"""
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import glob
import os
import shutil




####################################################
#   Identifying single/multiple images
##################################################


def rail_image_classifier(images, optimizer='rmsprop'):
    """
    images: a single or a list of images with their path (e.g '/home/wilson/'railroad.png')
    optimizer: model optimizer  
    """
    best_model = load_model('sequential_categorical.h5')    
    best_model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics=['accuracy'])
    
    img_width, img_height = 256, 256        #--- dimensions of our images
    
    for raw_img in images:
        img = image.load_img(raw_img, target_size=(img_width, img_height))
        img = image.img_to_array(img)/255.
        img = np.expand_dims(img, axis=0)
        
        pred = best_model.predict_classes(img)
        
        if pred[0] == 0:
            print(raw_img + ' is' + ' OTHER.')   
        elif pred[0] == 1:
            print(raw_img + ' is' + ' RAILROAD.')   
    
    return
    
    
images = ['other1.png',  'other2.png', 'other3.png', 'other4.png', 'rail1.png', 'rail2.png', 
          'rail3.png', 'rail4.png', 'rail5.png'] 




    
##############################################################
#  Separate images into different directory by prediction 
##############################################################
   
def railroad_classifier(dir_Dataset, dir_Railroad, dir_Other, filetype='png'):
    """
    Collected images are sent to the different directory by their prediction
    
    * dir_dataset: directory path of collected images 
    * dir_Railroad: directory path where images predicted as "railroad" will be 
    * dir_Other: directory path where images predicted as "other" will be
    * filetype: image file extension, e.g. .png, jpg...
    
    """ 
    
    ###=== Load the Best CNN model
    best_model = load_model('sequential_categorical.h5')
    best_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


    ###=== Classify railroad and other images using CNN model
    img_width, img_height = 256, 256        #--- dimensions of our images
    
    collected_img_path = glob.glob(dir_Dataset + '*' + '.' + filetype)
    
    num_image_classified = 0 
    
    for raw_img in collected_img_path:    
        img = image.load_img(raw_img, target_size=(img_width, img_height))
        img = image.img_to_array(img)/255.
        img = np.expand_dims(img, axis=0)
    
        pred = best_model.predict_classes(img)
        
        if pred[0] == 1:
            shutil.move(raw_img, dir_Railroad)            
        elif pred[0] == 0:
            shutil.move(raw_img, dir_Other)
            
        num_image_classified += 1
    
    ###=== the number of images processed 
    assert num_image_classified == len(collected_img_path), "all images are classified"
    print("The number of classified images: " + str(num_image_classified))
    
    return  
    

#dir_Dataset = '/home/wilson/Documents/AgileBeat/tile-data_0523/' 
#dir_Railroad = '/home/wilson/Documents/AgileBeat/pred_railroad/'
#dir_Other = '/home/wilson/Documents/AgileBeat/pred_other/'

