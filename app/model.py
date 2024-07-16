import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array


#labels
class_labels = ["fetal femur", "fetal abdomen", "fetal thorax", "fetal brain" , "maternal cervix"]

def load_model():
    """Loads and returns the fine-tuned model."""
    model = keras_load_model("C:/Users/Benna/OneDrive/Bureau/stage dete/model/my_model.keras")
    print("Model loaded")
    return model

def prepare_image(image, target):
    """Resize the input image and preprocess it."""
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # Assuming your model uses the same preprocessing as ImageNet models
    image = image / 255.0  # Normalize image if needed (adjust based on your model's requirements)
    return image

def predict(image, model):
    """Predict the class of the image using the model."""
    results = model.predict(image)
    top_indices = results[0].argsort()[-2:][::-1]  # Get top 2 predictions
    response = [
        {"class": class_labels[i], "score": float(round(results[0][i], 3))} for i in top_indices
    ]
    return response
