# -*- coding: utf-8 -*-
"""
Created on Mon May 10 01:55:50 2021

@author: Loai
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn

import cv2                  
from tqdm import tqdm
import os                   

import os
# print(os.listdir('./d'))


X=[]
Z=[]
IMG_SIZE=64
Apple___Apple_scab='./dataset_Segmentation/Apple___Apple_scab'
Apple___Black_rot='./dataset_Segmentation/Apple___Black_rot'
Apple___Cedar_apple_rust='./dataset_Segmentation/Apple___Cedar_apple_rust'
Apple___healthy='./dataset_Segmentation/Apple___healthy'


def assign_label(img,CropDisease):
    return CropDisease

def make_train_data(CropDisease,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,CropDisease)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
make_train_data('scab',Apple___Apple_scab)
print(len(X))

make_train_data('rot',Apple___Black_rot)
print(len(X))

make_train_data('rust',Apple___Cedar_apple_rust)
print(len(X))

make_train_data('healthy',Apple___healthy)
print(len(X))




le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,4)
X=np.array(X)
X=X/255.

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)



model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =32, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
# model.add(Activation('relu'))
model.add(Dense(4, activation = "softmax"))

batch_size=32
epochs=20

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1,factor=0.1)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              validation_data = (x_test,y_test),
                              verbose = 1, 
                              steps_per_epoch=x_train.shape[0] // batch_size)

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy Apple')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)
y_test=np.argmax(y_test,axis=1)



from sklearn.metrics import classification_report ,f1_score, roc_auc_score, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, pred_digits)
f1 = f1_score(y_test, pred_digits,average=None)
f3 = f1_score(y_test, pred_digits,average='macro')


print(classification_report(y_test, pred_digits,
                            target_names = ['scab (Class 0)',
                                            'rot (Class 1)', 
                                            'rust (Class 2)',
                                            'healthy (Class 3)']))