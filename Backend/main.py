from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image   
import io

from .model import DiseaseDiagnosisModel, device, num_classes
app = FastAPI(title= "AI X-rayScan API")

# Enable CORS for all origins (for development purposes)
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
class_names = ['Normal', 'Pneumonia', 'Tuberculosis']  # Example class names
num_classes = len(class_names)

model = DiseaseDiagnosisModel(num_classes)
model.load_state_dict(torch.load("disease_diagnosis_model.pth", map_location=device))
model.to(device)    
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((128, 128)),      
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict_disease(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read() # Read the uploaded file
    image = Image.open(io.BytesIO(image_data)).convert("RGB") # Convert to RGB
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device 
    
    # Make prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(input_tensor) # Get model outputs
        _, predicted = torch.max(outputs, 1) # Get the index (eg: 0(normal), 1(pneumonia), 2(tuberculosis)) of the highest score
        predicted_class = class_names[predicted.item()] # Map index to class name
    
    return {"prediction": predicted_class} # Return the predicted class as JSON response

