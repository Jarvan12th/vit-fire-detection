import io
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import requests
from pydantic import BaseModel

app = FastAPI()

# Load the model and feature extractor
model = AutoModelForImageClassification.from_pretrained("./vit_fire_detection_model/")
extractor = AutoFeatureExtractor.from_pretrained("./vit_fire_detection_model/")


# Define the request and response models
class ImageURL(BaseModel):
    image_url: str


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict_url", response_model=PredictionResponse)
def predict_url(request: ImageURL):
    # Fetch the image from the provided URL
    response = requests.get(request.image_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Convert the response content to a PIL Image
    image = Image.open(io.BytesIO(response.content))

    # Process the image with the feature extractor
    inputs = extractor(image, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the prediction
    predicted_class_idx = outputs.logits.argmax().item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return {"prediction": predicted_class}


@app.post("/predict_file", response_model=PredictionResponse)
def predict_file(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))

    # Process the image with the feature extractor
    inputs = extractor(image, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the prediction
    predicted_class_idx = outputs.logits.argmax().item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return {"prediction": predicted_class}


# The following code is for testing the API locally
if __name__ == "__main__":
    result = predict_url(
        ImageURL(
            image_url="https://test-bucket-jarvan.s3.us-west-2.amazonaws.com/fire-building.jpeg"
        )
    )
    print(result)
