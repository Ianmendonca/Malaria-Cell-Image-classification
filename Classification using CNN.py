import os
import cv2
import numpy as np

from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import precision_score, recall_score

''' Read the data from the folder and resize/ process the images as üper your requirement'''

size = 150
def data_prepocessing(path):
    images = []
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('.png'):
            img_path = os.path.join(path,file_name)
            img = cv2.imread(img_path)
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_RGB,(size,size))
            images.append(resized_img)
    return np.array(images)

infected_images = data_prepocessing("Path to your respective directory//Parasitized")
noninfected_images= data_prepocessing("Path to your respective directory//Uninfected")

# create labels for the data 1 = infected, 0 = non infected
infected_labels = np.ones(len(infected_images))
noninfected_labels = np.zeros(len(noninfected_images))

# concatenate the data
X = np.concatenate((infected_images, noninfected_images))
y = np.concatenate((infected_labels, noninfected_labels))

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# model

INPUT_SHAPE = (size, size, 3)   #change to (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))  

# specify the hyperparameters for the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# print the model summary 
# print(model.summary())  

# train the model 

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 64,
                         epochs = 5,
                         shuffle = False
                     )

#validate the model
loss, accuracy = model.evaluate(X_test, y_test)

print("Validation loss:", loss)
print("Validation accuracy:", accuracy)
