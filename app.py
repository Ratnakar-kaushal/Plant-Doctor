import os

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from model import PlantDiseaseCNN

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = PlantDiseaseCNN(num_classes=3)
model.load_state_dict(torch.load('custom_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Setup device
device = torch.device("cpu")
model.to(device)

# Class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded image
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Open and transform the image
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # Check if confidence is below 80%, classify as non-plant or unclear image
    if confidence < 0.80:
        prediction = "Unclear image or not a plant"
        confidence = 100 - confidence * 100
    else:
        prediction = class_names[pred_class]
        confidence = confidence * 100

    # Render result
    return render_template('index.html', 
                           prediction=prediction,
                           confidence=f"{confidence:.2f}%",
                           image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
