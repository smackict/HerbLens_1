from tensorflow.keras.models import model_from_json
import numpy as np
import os
import tensorflow as tf

classes = ['bellflower','common_daisy','rose','sunflower']

json_file = open('/src/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print('JSON model loaded from disk!')
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/src/model/cnn_weights.best.weights.h5")
print("Loaded weights from disk!")


def predict_image(img_path):
  img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
  img = img.resize((224,224))
  img = np.array(img)
  img = img.reshape(1,224,224,3)
  img = img/255.0
  predicted = loaded_model.predict(img)

  print(predicted)

  predictedClass = np.argmax(predicted)
  print(predictedClass)
  print(classes[predictedClass])

  return classes[predictedClass]

  