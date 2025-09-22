from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
import uvicorn
import logging
from typing import List
from schema.user_input import preprocess_image
from schema.output import postprocess_output, tensor_to_base64
from UNet import UNet 

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ConditionalUNet Polygon Coloring API",
    description="A web API for coloring polygon outlines using your trained ConditionalUNet model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://conditionalunet.netlify.app",  
        "http://localhost:8000",              # For local development 
        "http://127.0.0.1:8000", 
        "http://127.0.0.1:3000",
        "http://localhost:3000"           # For local development 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None

# Color mapping based on model training
COLOR_MAP = {
    0: "red",
    1: "blue", 
    2: "green",
    3: "yellow",
    4: "purple",
    5: "orange",
    6: "cyan",
    7: "magenta"
}

COLOR_NAME_TO_IDX = {v: k for k, v in COLOR_MAP.items()}


def load_model():
    """Load your trained ConditionalUNet model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize model with the same parameters used during training
        model = UNet(num_colors=8).to(device)  
        
        # Load trained model weights
        model_path = "model/best_model.pth" 
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            logger.info(f"Model successfully loaded from {model_path}")
            logger.info(f"Model running on {device}")
            
           
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    logger.info("Loading ConditionalUNet model...")
    load_model()
    logger.info("Server startup complete!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ConditionalUNet Polygon Coloring API",
        "description": "Upload polygon outlines and get them colored in your desired color",
        "version": "1.0.0",
        "model_info": {
            "architecture": "ConditionalUNet",
            "supported_colors": len(COLOR_MAP),
            "input_size": "128x128",
            "device": str(device) if device else "not loaded"
        },
        "endpoints": {
            "POST /colorize": "Upload polygon outline and specify color to get colored version",
            "GET /colors": "Get list of all supported colors",
            "GET /health": "Check API and model health",
            "GET /model-info": "Get detailed model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "api": "running",
        "model_loaded": model_loaded,
        "device": str(device) if device else "not initialized",
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }

@app.get("/colors")
async def get_supported_colors():
    """Get list of all supported colors with their indices"""
    return {
        "supported_colors": COLOR_MAP,
        "color_names": list(COLOR_NAME_TO_IDX.keys()),
        "total_colors": len(COLOR_MAP),
        "usage": "Use any color name in the 'color' parameter when calling /colorize"
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_architecture": "ConditionalUNet",
        "parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "input_channels": 3,
        "output_channels": 3,
        "supported_colors": len(COLOR_MAP),
        "device": str(device),
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2)  # Approximate size in MB
    }



@app.post("/colorize")
async def colorize_polygon(
    file: UploadFile = File(..., description="Polygon outline image (PNG/JPG)"),
    color: str = Form(..., description="Target color name (e.g., 'red', 'blue')")
):
    """
    Colorize a polygon outline with the specified color
    
    **Parameters:**
    - **file**: Image file containing the polygon outline (PNG, JPG, JPEG supported)
    - **color**: Target color name. Must be one of: red, blue, green, yellow, purple, orange, cyan, magenta
    
    **Returns:**
    - JSON response with the colored image as base64 string
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    # Validate color input
    color_lower = color.lower().strip()
    if color_lower not in COLOR_NAME_TO_IDX:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported color: '{color}'. Supported colors: {list(COLOR_NAME_TO_IDX.keys())}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file (PNG, JPG, JPEG)."
        )
    
    try:
        # Read and process the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        input_tensor = preprocess_image(image).to(device)
        
        # Get color index
        color_idx = torch.tensor([COLOR_NAME_TO_IDX[color_lower]]).to(device)
        
        # Generate colored polygon
        with torch.no_grad():
            colored_output = model(input_tensor, color_idx)
        
        # Convert output to base64
        output_base64 = tensor_to_base64(colored_output.cpu())
        
        return JSONResponse(content={
            "success": True,
            "message": f"Polygon successfully colored in {color_lower}",
            "input_info": {
                "filename": file.filename,
                "original_size": f"{image.size[0]}x{image.size[1]}",
                "processed_size": "128x128"
            },
            "color_applied": color_lower,
            "colored_image": f"data:image/png;base64,{output_base64}"
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the image: {str(e)}"
        )

@app.post("/colorize-batch")
async def colorize_polygon_batch(
    file: UploadFile = File(..., description="Polygon outline image"),
    colors: List[str] = Form(..., description="Comma-separated list of colors (e.g., 'red,blue,green')")
):
    """
    Colorize a polygon outline with multiple colors in one request
    
    **Parameters:**
    - **file**: Image file containing the polygon outline
    - **colors**: Comma-separated color names (e.g., "red,blue,green")
    
    **Returns:**
    - JSON response with multiple colored versions
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Parse colors
    color_list = [c.strip().lower() for c in colors]
    
    # Validate all colors
    invalid_colors = [c for c in color_list if c not in COLOR_NAME_TO_IDX]
    if invalid_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid colors: {invalid_colors}. Supported: {list(COLOR_NAME_TO_IDX.keys())}"
        )
    
    try:
        # Process image once
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor = preprocess_image(image).to(device)
        
        results = {}
        
        # Generate colored versions for each color
        with torch.no_grad():
            for color_name in color_list:
                color_idx = torch.tensor([COLOR_NAME_TO_IDX[color_name]]).to(device)
                colored_output = model(input_tensor, color_idx)
                output_base64 = tensor_to_base64(colored_output.cpu())
                results[color_name] = f"data:image/png;base64,{output_base64}"
        
        return JSONResponse(content={
            "success": True,
            "message": f"Polygon colored in {len(color_list)} colors",
            "input_filename": file.filename,
            "colors_generated": color_list,
            "colored_images": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")
    

@app.post("/colorize-true-batch")
async def colorize_true_batch(
    files: List[UploadFile] = File(..., description="A list of polygon outline images"),
    colors: List[str] = Form(..., description="A list of colors for each image")
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    color_list = [c.strip().lower() for c in colors]

    if len(files) != len(color_list):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatch: Received {len(files)} files but {len(color_list)} colors."
        )


    invalid_colors = [c for c in color_list if c not in COLOR_NAME_TO_IDX]
    if invalid_colors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid colors found: {invalid_colors}"
        )

    # --- 2. Preprocess all images and colors ---
    image_tensors = []
    color_indices = []

    try:
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")
            
            # Preprocess each image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            tensor = preprocess_image(image).squeeze(0) 
            image_tensors.append(tensor)

            # Collect corresponding color index
            color_name = color_list[i]
            color_indices.append(COLOR_NAME_TO_IDX[color_name])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or processing files: {e}")


    # --- 3. Stack into a single batch ---
    batch_image_tensor = torch.stack(image_tensors).to(device)
    batch_color_indices = torch.tensor(color_indices).to(device)


    # --- 4. Perform Batch Inference ---
    with torch.no_grad():
        output_batch = model(batch_image_tensor, batch_color_indices)


    # --- 5. Post-process and collect results ---
    results = []
    for i in range(len(files)):
        single_output_tensor = output_batch[i]
        
        output_base64 = tensor_to_base64(single_output_tensor.cpu())
        
        results.append({
            "filename": files[i].filename,
            "color_applied": color_list[i],
            "colored_image": f"data:image/png;base64,{output_base64}"
        })

    return JSONResponse(content={
        "success": True,
        "message": f"Successfully processed {len(files)} images in a batch.",
        "results": results
    })

os.makedirs("models", exist_ok=True)
os.makedirs("temp", exist_ok=True)

if __name__ == "__main__":
    logger.info("Starting ConditionalUNet FastAPI Server...")
    logger.info("API Documentation will be available at:")
    logger.info("Alternative docs at:")
    
    uvicorn.run(
        "app:app",  
        host="0.0.0.0",
        port=8000,
        reload=True, 
        log_level="info"
    )