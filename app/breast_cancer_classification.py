from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image
import logging
import io

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_b(model_path):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def transform_image_b(image):
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformation(image)

def predict_image_b(image_bytes, model):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform_image(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            prediction = torch.sigmoid(output).item()
        return prediction
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise
