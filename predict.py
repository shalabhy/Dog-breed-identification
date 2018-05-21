import pandas as pd
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.transform import resize
import keras
from keras.models import Model



test_path = 'test'
sample_file_path = 'sample_subvmission.csv'

sub = pd.read_csv('sample_submission.csv')
test_files = os.listdir(test_path)

test_temp = np.zeros((1,224,224,3))

rnet =keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
rnet.layers.pop()
top_model = keras.Sequential()
top_model.add(keras.layers.Dense(120,activation='softmax',input_shape=(2048,)))
rnet = Model(rnet.input,rnet.layers[-1].output)

full_model = Model(input=rnet.input,output=top_model(rnet.output))


full_model.load_weights("model.h5")
for i in range(len(test_files)):
  img = plt.imread(test_path+'/'+test_files[i])
  test_temp[0,:] = resize(img,(224,224))

  sub.iloc[i,1:] = full_model.predict(test_temp).reshape(-1,)
  if i%100 == 0:
    print('{} iterations done'.format(i))



sub.to_csv("resnet_submission.csv", index = False)