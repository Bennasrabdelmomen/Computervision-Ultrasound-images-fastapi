import simplejson as json
import tensorflow as tf
from tensorflow import keras

with open("C:/Users/Benna/OneDrive/Bureau/stage dete/to irm or not to irm/model/model_arch.json", 'r') as json_file:
    model_json = json.load(json_file)
loaded_model = tf.keras.models.model_from_json(model_json)

with open("C:/Users/Benna/OneDrive/Bureau/stage dete/to irm or not to irm/model/m.json", 'w') as json_file:
    json_file.write(json.dumps(json.loads(loaded_model), indent=4))
