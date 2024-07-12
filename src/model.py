import keras
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes = ['bellflower', 'common_daisy', 'rose', 'sunflower']

# loaded_model = keras.models.load_model("/opt/render/project/src/src/model/flowers_model.keras")

with open('/opt/render/project/src/src/model/model.json', 'r') as r:
  model_json = r.read()

# loaded_model_json = model_json.read()
# json_file.close()
print('JSON model loaded from disk!')
loaded_model = model_from_json(model_json)

loaded_model.load_weights(
    "/opt/render/project/src/src/model/cnn_weights.best.weights.h5")
print("Loaded weights from disk!")


def predict_image(img_path):
  img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
  img = img.resize((224, 224))
  img = np.array(img)
  img = img.reshape(1, 224, 224, 3)
  img = img / 255.0
  predicted = loaded_model.predict(img)

  print(predicted)

  predictedClass = np.argmax(predicted)
  print(predictedClass)
  print(classes[predictedClass])

  return classes[predictedClass]
