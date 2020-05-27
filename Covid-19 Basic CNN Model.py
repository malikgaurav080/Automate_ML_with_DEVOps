#!/usr/bin/env python
# coding: utf-8

# # Covid-19

# Let's first import the packages necessary for the task and load the dataset. 

# In[1]:


import os
from os.path import join
import numpy as np
import pandas as pd
import random
import glob
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn .preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot,plot_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D,Dropout,Flatten,Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array


# Setting the image dimensions.

# In[2]:


IMG_W = 150
IMG_H = 150
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
lbls = list(map(str, range(NB_CLASSES)))


# In[3]:


DATASET_DIR = "C:/Users/GAURAV's LAPPY/Desktop/MLOps_project/dataset/"


# Let's have a quick of data.

# In[4]:



normal_images = []
for img_path in glob.glob(DATASET_DIR + '/normal/*'):
    normal_images.append(mpimg.imread(img_path))
    

covid_images = []
for img_path in glob.glob(DATASET_DIR + '/covid/*'):
    covid_images.append(mpimg.imread(img_path))


# Now, applying the model.

# In[5]:


classifier = Sequential()

classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu',
                      input_shape = (IMG_W,IMG_H,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units = 2, activation = 'softmax'))
classifier.compile(optimizer = Adam(lr=0.001),loss='categorical_crossentropy', metrics = ['accuracy'])


# In[6]:


classifier.summary()


# In[7]:


# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        #featurewise_center=False,  # set input mean to 0 over the dataset
        #samplewise_center=False,  # set each sample mean to 0
        #featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #samplewise_std_normalization=False,  # divide each input by its std
        #zca_whitening=False,  # apply ZCA whitening
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        #zoom_range = 0.1, # Randomly zoom image 
        #width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #horizontal_flip=True,  # randomly flip images
        #vertical_flip=False,  # randomly flip images
        #preprocessing_function=preprocess_image,
        rescale = 1./255,
        validation_split = 0.3) 
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=4,
    class_mode='categorical',
    subset='training')
validation_generator = datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=(IMG_H, IMG_W),
    batch_size=4,
    class_mode='categorical',
    shuffle= False,
    subset='validation')


# In[8]:


es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 10)
mc = ModelCheckpoint('F:/cnn_covid_pred.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)


# In[9]:


epochs = 1
steps_per_epoch = 2


# In[ ]:


history = classifier.fit_generator(train_generator,
                         steps_per_epoch = steps_per_epoch,
                         epochs=epochs,
                         callbacks = [es, mc],
                         workers=4,
                         validation_data = validation_generator,
                         validation_steps = 10)


# In[ ]:

print(history.history)
exit()
training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]
print("training_accuracy ", training_accuracy)
print("validation_accuracy ", validation_accuracy)


# In[ ]:


print("training_loss", history.history['loss'][-1])
print("validation_loss", history.history['val_loss'][-1])


# In[ ]:


for i in range(4):
    if training_accuracy < 0.95 :
        epochs+=5
        steps_per_epoch+=1
        history = classifier.fit_generator(train_generator,
                             steps_per_epoch = steps_per_epoch,
                             epochs=epochs,
                             callbacks = [es, mc],
                             workers=4,
                             validation_data = validation_generator,
                             validation_steps = 10)
        training_accuracy = history.history['accuracy'][-1]
        validation_accuracy = history.history['val_accuracy'][-1]
        print("training_accuracy ", training_accuracy)
        print("validation_accuracy ", validation_accuracy)
    else:
        break


# In[ ]:


if training_accuracy < 0.95 :
    print("you dont achieve beter accuracy, now you have to add more convolution layers")
else :
    print("you achieve better accuracy")


# In[10]:


if training_accuracy < 0.95 :
    classifier = Sequential()
    classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu',
                      input_shape = (IMG_W,IMG_H,3)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #here add one more convolution layer
    classifier.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.20))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    classifier.compile(optimizer = Adam(lr=0.001),loss='categorical_crossentropy', metrics = ['accuracy'])
    
    classifier.summary()
    
    history = classifier.fit_generator(train_generator,
                         steps_per_epoch = steps_per_epoch,
                         epochs=epochs,
                         callbacks = [es, mc],
                         workers=4,
                         validation_data = validation_generator,
                         validation_steps = 10)
    
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    print("training_accuracy ", training_accuracy)
    print("validation_accuracy ", validation_accuracy)
else:
    pass


# In[ ]:


if training_accuracy > 0.95 :
    pass
else:
    classifier = Sequential()
    classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu',
                      input_shape = (IMG_W,IMG_H,3)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    #here add one more convolution layer
    classifier.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    #here add one more convolution layer
    classifier.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.20))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    classifier.compile(optimizer = Adam(lr=0.001),loss='categorical_crossentropy', metrics = ['accuracy'])
    
    classifier.summary()
    
    history = classifier.fit_generator(train_generator,
                         steps_per_epoch = steps_per_epoch,
                         epochs=epochs,
                         callbacks = [es, mc],
                         workers=4,
                         validation_data = validation_generator,
                         validation_steps = 10)
    
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    print("training_accuracy ", training_accuracy)
    print("validation_accuracy ", validation_accuracy)


# In[ ]:


if training_accuracy <0.95 :
    print("something went wrong plzz... contact to developer")

