# ConditionalUNet Polygon Coloring 🎨

An end-to-end machine learning project that uses a custom-built Conditional UNet to colorize polygon outlines with user-specified colors. The system features a PyTorch-based deep learning model, containerized FastAPI backend, and a modern web interface.

## 🌟 Project Overview

This project demonstrates a complete ML pipeline from model development to production deployment:
- **Custom UNet Architecture**: Built from scratch in PyTorch with color conditioning
- **Containerized API**: FastAPI backend deployed on AWS EC2 using Docker
- **Web Application**: Modern frontend deployed on Netlify with real-time inference
- **End-to-End Pipeline**: From data preprocessing to model serving

## 🏗️ Architecture

```
Frontend (Netlify) → API Gateway (Netlify Functions) → FastAPI (AWS EC2) → UNet Model → Response
```

### Model Architecture
- **Conditional UNet**: Custom implementation with color embedding integration
- **Input**: 128x128 RGB polygon outlines + color index
- **Output**: 128x128 RGB colored polygons
- **Colors Supported**: 8 distinct colors (red, blue, green, yellow, purple, orange, cyan, magenta)

## 🚀 Live Demo

- **Web Application**: [https://conditionalunet.netlify.app](https://conditionalunet.netlify.app)
- **API Documentation**: Available at the deployed API endpoint `/docs`

## 📁 Project Structure

```
unet-web-app/
├── 📂 model/                    # Model architecture and components
│   ├── UNet.py                  # Main UNet implementation
│   ├── UNet_parts.py           # UNet building blocks
│   └── best_model.pth          # Trained model weights (download required)
├── 📂 schema/                   # Data processing utilities
│   ├── user_input.py           # Input preprocessing
│   └── output.py               # Output postprocessing
├── 📂 netlify/functions/       # Serverless proxy functions
│   └── proxy.js                # CORS proxy for API calls
├── 📂 training/                # Training notebooks and scripts
│   └── [See: https://github.com/jagapathi20/ConditionalUNet]
├── app.py                      # FastAPI application
├── dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── index.html                  # Frontend application
└── README.md                   # Project documentation
```

## 🧠 Model Details

### UNet Architecture
- **Encoder**: 4 down-sampling blocks with double convolutions
- **Decoder**: 4 up-sampling blocks with skip connections
- **Bottleneck**: Feature compression layer
- **Color Conditioning**: Embedding layer that integrates color information throughout the network

### Key Features
- Skip connections for better feature preservation
- Batch normalization for training stability
- Color embedding integration for conditional generation
- Optimized for 128x128 image resolution

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI
- **ML Library**: PyTorch
- **Image Processing**: Pillow, OpenCV
- **Deployment**: Docker + AWS EC2
- **API Features**: CORS support, batch processing, health checks

### Frontend
- **Technology**: Vanilla JavaScript + HTML/CSS
- **Styling**: Modern CSS with gradients and animations
- **Features**: Drag-and-drop upload, real-time preview, batch processing
- **Deployment**: Netlify with serverless functions

### Infrastructure
- **Containerization**: Docker
- **Cloud**: AWS EC2 for API hosting
- **CDN**: Netlify for frontend delivery
- **Registry**: Docker Hub for image storage

## 📦 Installation & Setup

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/jagapathi20/ConditionalUNet_web_app
cd unet-web-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**
```bash
# Download the trained model weights from Google Drive
# Link: https://drive.google.com/file/d/11t5PHTNSSkouYGKjBtNVMz3dCqp7AOpK/view
# Save as: model/best_model.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11t5PHTNSSkouYGKjBtNVMz3dCqp7AOpK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11t5PHTNSSkouYGKjBtNVMz3dCqp7AOpK" -O model/best_model.pth && rm -rf /tmp/cookies.txt
```

4. **Run the API server**
```bash
python app.py
# Or using uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. **Serve the frontend**
```bash
# Open index.html in a browser or use a local server
python -m http.server 3000
```

### Docker Deployment

1. **Build the image**
```bash
docker build -t unet-colorizer .
```

2. **Run the container**
```bash
docker run -p 8000:8000 unet-colorizer
```

3. **Deploy to cloud**
```bash
# Push to Docker Hub
docker tag unet-colorizer your-username/unet-colorizer
docker push your-username/unet-colorizer
```

## 🎯 API Endpoints

### Core Endpoints
- `GET /` - API information and health status
- `POST /colorize` - Single image colorization
- `POST /colorize-batch` - Multiple colors for one image
- `POST /colorize-true-batch` - Batch processing multiple images
- `GET /colors` - List supported colors
- `GET /health` - System health check

### Example Usage

```javascript
// Single image colorization
const formData = new FormData();
formData.append('file', imageFile);
formData.append('color', 'red');

const response = await fetch('/colorize', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.colored_image); // Base64 image data
```

## 🎨 Supported Colors

The model supports 8 distinct colors:
- **Red** (#ef4444)
- **Blue** (#3b82f6)
- **Green** (#10b981)
- **Yellow** (#f59e0b)
- **Purple** (#8b5cf6)
- **Orange** (#f97316)
- **Cyan** (#06b6d4)
- **Magenta** (#ec4899)

## 🏋️ Training Details

### Complete Training Repository
**Full training code, datasets, and experiments are available at:**
📚 **[ConditionalUNet Training Repository](https://github.com/jagapathi20/ConditionalUNet)**

This repository contains:
- Complete training pipeline and notebooks
- Dataset preparation and augmentation code
- Experiment tracking and model evaluation
- Training logs and performance metrics
- Data preprocessing utilities

### Dataset
- **Custom polygon dataset** with outline-colored pairs
- **Data Augmentation**: Rotation, scaling, color variation
- **Training Details**: Available in the training repository

### Training Process
- **Loss Function**: Custom combination of MSE and perceptual loss
- **Optimizer**: Adam with learning rate scheduling
- **Training Platform**: Google Colab with GPU acceleration
- **Model Checkpoints**: Best model saved based on validation loss

### Model Performance
- **Training Metrics**: Detailed in training repository
- **Model Size**: ~45MB
- **Inference Speed**: Optimized for real-time processing

## 🚀 Deployment Architecture

### AWS EC2 Deployment
- **Instance Type**: [Instance details]
- **Container Runtime**: Docker
- **Port Mapping**: 8000:8000
- **Security Groups**: Configured for API access

### Netlify Frontend
- **Build Process**: Static site deployment
- **Serverless Functions**: CORS proxy for API calls
- **Custom Domain**: Available on request

### CI/CD Pipeline
- **Container Registry**: Docker Hub
- **Deployment**: Manual deployment to EC2
- **Monitoring**: Basic health checks

## 🔧 Development Features

### API Features
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Comprehensive error responses
- **Input Validation**: File type and color validation
- **Batch Processing**: Multiple processing modes
- **Health Monitoring**: System status endpoints

### Frontend Features
- **Responsive Design**: Mobile and desktop optimized
- **Drag-and-Drop**: Intuitive file upload
- **Real-time Preview**: Instant image preview
- **Batch Mode**: Multiple file processing
- **Download Support**: Direct image download

## 📈 Performance Metrics

### Model Performance
- **Inference Time**: ~200-500ms per image
- **Memory Usage**: ~2GB for model loading
- **Batch Processing**: Up to 10 images simultaneously

### System Performance
- **API Response Time**: ~1-3 seconds
- **Concurrent Users**: Tested up to [number] users
- **Uptime**: [Monitoring stats]

## 🔮 Future Enhancements

### Model Improvements
- [ ] Support for more colors and custom color palettes
- [ ] Higher resolution output (256x256, 512x512)
- [ ] Style transfer capabilities
- [ ] Real-time processing optimization

### System Enhancements
- [ ] Auto-scaling deployment
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Advanced monitoring and logging

### Feature Additions
- [ ] User account system
- [ ] Processing history
- [ ] Bulk download functionality
- [ ] API rate limiting

## 📊 Model Weights & Training Resources

### Pre-trained Model
- **Model Weights**: [Download from Google Drive](https://drive.google.com/file/d/11t5PHTNSSkouYGKjBtNVMz3dCqp7AOpK/view)
- **Model Size**: ~45MB
- **Format**: PyTorch state dict (.pth)
- **Installation**: Download and place in `model/best_model.pth`

### Training Resources
- **Complete Training Code**: [ConditionalUNet Repository](https://github.com/jagapathi20/ConditionalUNet)
- **Training Notebooks**: Google Colab notebooks with full pipeline
- **Dataset & Preprocessing**: Available in training repository
- **Experiment Logs**: Training history and model evaluation



---

*This project demonstrates end-to-end ML system design, from custom model architecture to production deployment. Perfect for understanding modern ML operations and deployment strategies.*