
import keras
from keras.models import Model
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os 
from random import randrange

train_data_path = 'train'
label_data_path = 'labels.csv'


rnet =keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
rnet.layers.pop()

for layer in rnet.layers:
  layer.trainable = False

train_imgs = os.listdir(train_data_path)

top_model = keras.Sequential()
top_model.add(keras.layers.Dense(120,activation='softmax',input_shape=(2048,)))
rnet = Model(rnet.input,rnet.layers[-1].output)

full_model = Model(input=rnet.input,output=top_model(rnet.output))
full_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

labels_data = pd.read_csv(label_data_path)

encode = LabelEncoder()
onehot = OneHotEncoder(sparse = False)

y = onehot.fit_transform(encode.fit_transform(labels_data['breed']).reshape(-1,1))

train_temp = np.zeros((1000,224,224,3))
targets_temp = np.zeros((1000,120))

#training begins
for j in range(100):
  if j ==0:
    for i in range(1000):
      a = randrange(0,10222)
      img = plt.imread('images_data/train/'+train_imgs_path[a])
      train_temp[i,:] = resize(img,(224,224))
      targets_temp[i] = y[a] 
    #full_model.load_weights("model.h5")
    full_model.fit(train_temp, targets_temp, epochs=1, batch_size=50,  verbose=2)
    full_model.save_weights("model.h5")  
  else:
    for i in range(1000):
      a = randrange(0,10222)
      img = plt.imread('images_data/train/'+train_imgs_path[a])
      train_temp[i,:] = resize(img,(224,224))
      targets_temp[i] = y[a] 
    full_model.load_weights("model.h5")
    full_model.fit(train_temp, targets_temp, epochs=1, batch_size=50,  verbose=2)
    full_model.save_weights("model.h5") 
    
