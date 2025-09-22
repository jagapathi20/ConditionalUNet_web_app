import torchvision.transforms as transforms
from PIL import Image
import torch


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the input image for the model"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    TRANSFORM = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust if needed
    ])
    
    # Apply transformations
    tensor = TRANSFORM(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor