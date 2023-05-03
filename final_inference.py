import numpy as np
import os
import PIL
import PIL.Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf
import datetime
from numpy import random
import json
#===============================================================================================
# load pre-trained model (check tf for avaiable models)
# we will use InceptionV3 here
base_model = InceptionV3(weights='imagenet', include_top=False)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.00005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',

)

#
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#==============================================================================================
# add classification layers on top of the pre-trained model

# add a global spatial average pooling layer
def create_model():
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(64, activation='relu')(x)
    # x = layers.Dropout(.2)(x)
    x = Dense(32, activation='relu')(x)
    # x = layers.Dropout(.2)(x)
    predictions = Dense(6, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False
    # this is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model
#======================================== compile model ===============================================
def compile_model(model,optimizer,loss):
  model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

save_model_path = 'cp-0054.ckpt'     # insert the saved pre-trained model address here
prediction_model  = create_model()

prediction_model.load_weights(save_model_path)
prediction_model = compile_model(prediction_model,optimizer,loss)

#==============================================================================================
def create_samples_from_idx(facies_data,index=0):
  sample_idx = index
  # this should be converted into a new function / so the function should be classed insude the for loop
  image_data = facies_data[sample_idx:sample_idx+192]

  image = np.array(image_data)
  image = image/128
  image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
  image4d = np.expand_dims(image, axis=0)
  # print('shape of the created image is {}'.format(np.shape(image)))
  return image,image4d

#==============================================================================================
#Load the log

import json
with open('CRC2_1167_1169-5.json','r') as file:    # inset the save json file address here
  data=json.load(file)
#load image
print(len(data['image']))
print(len(data['Depth']))
def find_closest_value(vector,value):
    temp = [abs(i-value) for i in vector]
    indx = temp.index(min(temp))
    return vector[indx], indx
D= [float(i[1]) for i in np.array(data['Depth'])]

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def category_smoothing_by_line(my_array):
  for i in range(len(my_array)):
    # if i==540:
    #     continue
    temp = [0, 0, 0, 0, 0, 0]
    for j in my_array[i]:
      temp[j-1]+=1
    my_array[i] = np.argmax(temp)+1

  return my_array
#========================== evaluate json image ===============================================
inference_array = init_list_of_objects(len(D))
for i in np.arange(1,len(D)-192,1):
  temp_img,temp_img4d = create_samples_from_idx(data['image'],int(i))
  temp_list = prediction_model.predict(temp_img4d)
  indexes = np.arange(int(i),int(i)+192,1)
  if (i % 200) == 0:   print(f"line {i} of {len(D)-192} is proccessed")
  for j in range(len(indexes)):
      # inference_array[indexes[j]].append(np.argmax(temp_list)+1)
      inference_array[indexes[j]].append(temp_list)

my_array = inference_array[192:(len(D)-192-1)]
inference_depth = D[192:(len(D)-192-1)]

final_inference=[]
for i in range(len(my_array)):
    temp = [0, 0, 0, 0, 0, 0]
    for j in range(192):
        for k in range(6):
            temp[k] = temp[k] + my_array[i][j][0][k]
    final_inference.append(np.argmax(temp)+1)

out_put=[[inference_depth[0],final_inference[0]]]
temp = [inference_depth[0],final_inference[0]]
for i in range(len(final_inference)-1):
    if (final_inference[i] != final_inference[i+1]):
        temp = [inference_depth[i+1], final_inference[i+1]]
        out_put.append(temp)
out_put.append([inference_depth[-1],final_inference[-1]])

import pandas as pd
# print(Depth_array)
# df = pd.DataFrame(list(zip(Depth_arr,inference_arr)))
df = pd.DataFrame(list(np.array(out_put)))
with pd.ExcelWriter('inference_data.xlsx') as writer:
  df.to_excel(writer,sheet_name='Sheet1',index=False)

