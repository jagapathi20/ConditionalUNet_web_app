import torchvision.transforms as transforms
import base64
import torch
import io
from PIL import Image

import base64

INVERSE_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Denormalize
    transforms.ToPILImage()
])

def postprocess_output(output_tensor: torch.Tensor) -> Image.Image:
    """Convert model output back to PIL Image"""
    # Remove batch dimension and clamp values
    output = torch.clamp(output_tensor.squeeze(0), 0, 1)
    
    # Convert to PIL Image
    image = INVERSE_TRANSFORM(output)
    
    return image

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 encoded image string"""
    # Clamp and convert to PIL
    output = torch.clamp(tensor.squeeze(0), 0, 1)
    image = transforms.ToPILImage()(output)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    base64_str = base64.b64encode(buffer.getvalue()).decode()
    return base64_str
