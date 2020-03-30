import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

model = load_model('my_model11.h5')

def api(data1):
    predicted = model.predict(data1)
    return predicted

img= image.load_img("uploads/2018-10-19.png", color_mode="grayscale", target_size=(258,540,1))
x= image.img_to_array(img).astype("float32")
x=x/255.0
print(x)
print(x.shape)
x=x.reshape(1, 258, 540, 1)
            #indices = {1: 'Healthy', 0: 'Corona-Infected'}
result = api(x)
print("result=",result)
plt.imsave("static/image.jpg",result.reshape(258, 540))
            #predicted_class = np.asscalar(np.argmax(result, axis=1))
            #accuracy = round(result[0][predicted_class] * 100, 2)
            #label = indices[predicted_class]