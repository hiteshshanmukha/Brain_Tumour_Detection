# Brain Tumour Classifier

## Project Overview
This repository contains a comprehensive brain tumor classification system using various computer vision and deep learning approaches. The project implements multiple models for detecting and classifying brain tumors from MRI scans, including binary classification (tumor vs. no tumor) and multi-class classification (glioma, meningioma, no tumor).

## Repository Structure
```
Brain Tumour Classifier/
├── CNN+computer vision/
│   ├── final.ipynb
│   └── segmentation-canny.h5
├── Ensemble+Computer vision/
│   └── 1.ipynb
├── Unet/
│   ├── best_model.h5
│   └── CV_FINAL.ipynb
├── YOLO-V8/
│   ├── best.pt
│   └── brain-tumour-classification.ipynb
├── YOLO-V11/
│   ├── best.pt
│   └── mri-detection.ipynb
└── app/
    ├── app.py
    ├── best.pt
    └── v8.pt
```

## Key Features

- **Multiple Classification Approaches**: 
  - CNN-based models with computer vision techniques
  - Ensemble learning methods
  - U-Net architecture for tumor segmentation
  - YOLO-based object detection (V8 and V11)

- **Image Preprocessing Pipeline**:
  - Advanced denoising with BM3D filtering
  - Image normalization and standardization
  - Data augmentation techniques

- **Classification Tasks**:
  - Binary classification: Tumor vs. No tumor
  - Multi-class classification: Glioma, Meningioma, No tumor

- **Web Application**: Interactive interface for model inference

## Models Implemented

### 1. CNN with Computer Vision
Located in `CNN+computer vision/final.ipynb`, this approach combines conventional CNNs with computer vision techniques for improved tumor detection.

### 2. Ensemble Learning
Located in `Ensemble+Computer vision/1.ipynb`, this implementation uses ensemble methods combined with computer vision preprocessing for robust classification.

### 3. U-Net Segmentation
Located in `Unet/CV_FINAL.ipynb`, this implementation uses the U-Net architecture for precise tumor segmentation in MRI images.

### 4. YOLO Object Detection
- YOLO-V8: Located in `YOLO-V8/brain-tumour-classification.ipynb`
- YOLO-V11: Located in `YOLO-V11/mri-detection.ipynb`
  
These implementations treat tumor detection as an object detection problem, leveraging the power of YOLO models.

### 5. Web Application
Located in `app/app.py`, providing an interface to interact with the trained models.

## Installation and Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Brain-Tumour-Classifier.git
   cd Brain-Tumour-Classifier
   ```

2. Install required dependencies:
   ```
   pip install tensorflow opencv-python matplotlib numpy bm3d ultralytics
   ```

3. Download pre-trained models or train your own using the provided notebooks.

## Usage

### Running the Web Application
```
cd app
python app.py
```
This will start the web interface where you can upload MRI scans for tumor classification.

### Training Your Own Models
Each folder contains Jupyter notebooks that you can run to train the respective models:
- For CNN: Use `CNN+computer vision/final.ipynb`
- For Ensemble: Use `Ensemble+Computer vision/1.ipynb`
- For U-Net segmentation: Use `Unet/CV_FINAL.ipynb`
- For YOLO-V8: Use `YOLO-V8/brain-tumour-classification.ipynb`
- For YOLO-V11: Use `YOLO-V11/mri-detection.ipynb`

## Technologies Used

- TensorFlow/Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- BM3D (for image denoising)
- YOLO object detection framework
- U-Net architecture

## Dataset

The project uses MRI brain scan datasets that include:
- Binary classification dataset (tumor/no tumor)
- Multi-class dataset (glioma, meningioma, no tumor)

## Future Work

- Integration with more advanced deep learning architectures
- Expansion to detect additional brain pathologies
- Development of a more comprehensive medical diagnostic tool
- Deployment as a mobile application


