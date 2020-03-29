#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os  # iterating through directories & joining paths
import cv2  # open-cv library to do image operation
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


DATADIR = "datasets"


# In[3]:


CATEGORIES = ["Dog","Cat"]  

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)   # paths to cats and dogs directories
    class_num = CATEGORIES.index(category)  # adding label to data
    for img in os.listdir(path):     # iterating through every images
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)   # reading the image and converting into gray scale
        break
    break


# In[4]:


# since every photo is of different pixel size, converting all images to fixed pixel size
img_size=60
new_array = cv2.resize(img_array,(img_size, img_size))
plt.imshow(new_array, cmap="gray")
plt.show()


# In[5]:


training_data = []
img_size=60

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)   # paths to cats and dogs directories
        class_num = CATEGORIES.index(category)  # adding label to data
        for img in os.listdir(path):     # iterating through every images
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)   # reading the image and converting into gray scale
                new_array = cv2.resize(img_array,(img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()
            
            
            


# In[6]:


print(len(training_data))


# In[8]:


import random
random.shuffle(training_data)


# In[9]:



for cls in training_data[:10]:
    print(cls[0].shape)


# In[16]:


x=[]
y=[]
for features,label in training_data:
    x.append(features)
    y.append(label)
# print(x[0].shape)
print(x[0].shape)
X=np.array(x).reshape(-1, img_size, img_size, 1)  #-1 indicates to take all images,also takes shape of the image, and 1 indicate that image we want is gray scale
y=np.array(y)
# X.shape

print(X.shape)


# In[11]:


import pickle   # library to save the data

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[12]:


pickle_in = open("X.pickle","rb")   # reading the dataset
X1 = pickle.load(pickle_in)


# In[13]:


X1[0].shape
