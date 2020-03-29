#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout
import pickle

pickle_in = open("X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y=pickle.load(pickle_in)

# print(X.shape)  # (Total image,size,size,1)

X=X/255.0

model=Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:])) # convolutional net # window size of (3 x 3) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3))) # convolutional net # window size of (3 x 3) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# convolutional net is 2D therefore converting it to 1D by flattening it, the last fully connected neural net to label the patterns
model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size = 32, epochs=10,validation_split=0.1)



# In[3]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
path = "C:/Users/Dell/Desktop/abhi/practice/CatsAndDogs/random_test"
test=[]
img_size=60
def random_test():
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(img_size, img_size))
        x = np.array(new_array).reshape(-1,img_size, img_size, 1) 
        plt.imshow(new_array, cmap="gray")
        plt.show()
#         print(x.shape)
        x=x/255
#         test.append(x)
        ypred = model.predict(x)
        print(ypred)
#     print(test)
    
    
random_test()
    


# In[ ]:





# In[ ]:




