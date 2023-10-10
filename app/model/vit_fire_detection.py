# Load the model and processor
from transformers import ViTImageProcessor, AutoModelForImageClassification

processor = ViTImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")

# Save the model and processor to a directory
model.save_pretrained("./vit_fire_detection_model/")
processor.save_pretrained("./vit_fire_detection_model/")
