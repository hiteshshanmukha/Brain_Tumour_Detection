# 🧠 Brain Tumour Detection System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Project Overview

This repository presents a comprehensive **Brain Tumor Detection and Classification System** that leverages state-of-the-art computer vision and deep learning techniques. The system is designed to assist medical professionals in the early detection and classification of brain tumors from MRI scans.

### 🎯 Key Objectives:
- **Automated Detection**: Identify the presence of brain tumors in MRI scans
- **Multi-class Classification**: Classify tumors into specific types (Glioma, Meningioma, Pituitary)
- **Precision Medicine**: Provide accurate, reliable results to support medical diagnosis
- **Accessibility**: Offer an intuitive web interface for easy interaction

### 🏥 Medical Context:
Brain tumors are one of the most serious medical conditions requiring early and accurate diagnosis. This system aims to:
- Reduce diagnostic time
- Minimize human error
- Provide consistent analysis across different medical facilities
- Support radiologists with AI-powered insights

## 📁 Repository Structure

```
Brain_Tumour_Detection/
├── 📊 CNN+computer vision/          # Convolutional Neural Network implementation
│   ├── final.ipynb                 # Complete CNN training notebook
│   └── segmentation-canny.h5       # Saved CNN model with edge detection
├── 🔄 Ensemble+Computer vision/    # Ensemble learning approach
│   └── 1.ipynb                     # Ensemble model training notebook
├── 🎯 Unet/                        # U-Net segmentation implementation
│   ├── best_model.h5               # Saved U-Net model
│   └── CV_FINAL.ipynb              # U-Net training notebook
├── 📱 YOLO-V8/                     # YOLOv8 object detection
│   ├── best.pt                     # Pre-trained YOLOv8 model
│   └── brain-tumour-classification.ipynb
├── 🚀 YOLO-V11/                    # Latest YOLO implementation
│   ├── best.pt                     # Pre-trained YOLOv11 model
│   └── mri-detection.ipynb         # YOLOv11 training notebook
├── 🌐 app/                         # Web application
│   ├── app.py                      # Streamlit web interface
│   ├── best.pt                     # Production model
│   └── v8.pt                       # Alternative model
├── 📄 Group_27_report.pdf          # Detailed project report
└── 📝 README.md                    # This file
```

## ✨ Key Features

### 🔬 **Multiple Classification Approaches**
- **CNN-based Models**: Deep convolutional networks with advanced computer vision techniques
- **Ensemble Learning**: Combined predictions from multiple models for enhanced accuracy
- **U-Net Segmentation**: Precise pixel-level tumor boundary detection
- **YOLO Object Detection**: Real-time tumor localization using YOLOv8 and YOLOv11
- **Hybrid Approaches**: Integration of traditional CV with deep learning

### 🛠️ **Advanced Image Preprocessing Pipeline**
- **BM3D Denoising**: State-of-the-art noise reduction for cleaner MRI images
- **Edge Detection**: Canny edge detection for improved tumor boundary identification
- **Normalization & Standardization**: Consistent image preprocessing across datasets
- **Data Augmentation**: Rotation, scaling, and intensity variations for robust training
- **Histogram Equalization**: Enhanced contrast for better feature extraction

### 🎯 **Classification Capabilities**
- **Binary Classification**: Tumor presence detection (Tumor vs. No Tumor)
- **Multi-class Classification**: Specific tumor type identification
  - 🔴 **Glioma**: Most common primary brain tumor
  - 🟠 **Meningioma**: Typically benign tumor of brain membranes  
  - 🔵 **Pituitary Adenoma**: Tumor of the pituitary gland
  - ⚫ **Background/No Tumor**: Healthy brain tissue

### 💻 **Interactive Web Application**
- **Streamlit Interface**: User-friendly web application for medical professionals
- **Real-time Predictions**: Instant tumor detection and classification
- **Visual Results**: Bounding box annotations with confidence scores
- **Multiple Model Support**: Switch between different trained models

## 🤖 Models Implemented

### 1. 🧠 CNN with Computer Vision
**Location**: `CNN+computer vision/final.ipynb`

This implementation combines traditional Convolutional Neural Networks with advanced computer vision preprocessing techniques:
- **Architecture**: Multi-layer CNN with dropout regularization
- **Preprocessing**: Canny edge detection for enhanced tumor boundary detection
- **Features**: Handles complex MRI image patterns with high accuracy
- **Output**: Segmentation masks and classification results

### 2. 🔄 Ensemble Learning
**Location**: `Ensemble+Computer vision/1.ipynb`

Leverages the power of multiple models for robust predictions:
- **Strategy**: Combines predictions from multiple base learners
- **Voting Mechanism**: Weighted average of individual model outputs
- **Robustness**: Reduces overfitting and improves generalization
- **Performance**: Higher accuracy through model diversity

### 3. 🎯 U-Net Segmentation
**Location**: `Unet/CV_FINAL.ipynb`

Specialized architecture for precise medical image segmentation:
- **Architecture**: Encoder-decoder structure with skip connections
- **Specialty**: Excellent for biomedical image segmentation tasks
- **Output**: Pixel-level tumor masks with precise boundaries
- **Applications**: Surgical planning and radiation therapy guidance

### 4. 📱 YOLO Object Detection

#### YOLOv8 Implementation
**Location**: `YOLO-V8/brain-tumour-classification.ipynb`
- **Speed**: Real-time detection capabilities
- **Accuracy**: State-of-the-art object detection performance
- **Flexibility**: Handles multiple tumor types simultaneously

#### YOLOv11 Implementation  
**Location**: `YOLO-V11/mri-detection.ipynb`
- **Latest Technology**: Most recent YOLO architecture
- **Enhanced Features**: Improved accuracy and speed
- **Advanced Training**: Better convergence and stability

### 5. 🌐 Web Application
**Location**: `app/app.py`

Production-ready interface for medical professionals:
- **Framework**: Built with Streamlit for rapid deployment
- **Real-time Processing**: Instant image upload and analysis
- **Visual Feedback**: Bounding boxes with confidence scores and color-coded results
- **Model Integration**: Seamless integration with trained YOLO models

## ⚙️ System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: CUDA-compatible GPU recommended for training (RTX 3060+)
- **Storage**: At least 5GB free space for models and datasets
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)

### Software Requirements
- **Python**: 3.7 or higher (3.8+ recommended)
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **CUDA**: Version 11.0+ (if using GPU acceleration)

## 🚀 Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/hiteshshanmukha/Brain_Tumour_Detection.git
cd Brain_Tumour_Detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n brain-tumor python=3.8
conda activate brain-tumor

# Using venv
python -m venv brain-tumor-env
# On Windows
brain-tumor-env\Scripts\activate
# On macOS/Linux
source brain-tumor-env/bin/activate
```

### Step 3: Install Dependencies

#### Option 1: Using requirements.txt (Recommended)
```bash
# For full development environment
pip install -r requirements.txt

# For web application only  
pip install -r app/requirements.txt
```

#### Option 2: Manual Installation

##### Core Dependencies
```bash
pip install streamlit torch torchvision torchaudio
pip install ultralytics opencv-python pillow numpy matplotlib
pip install tensorflow keras scikit-learn pandas seaborn
```

##### Advanced Dependencies
```bash
pip install bm3d  # For advanced image denoising
pip install albumentations  # For data augmentation
pip install wandb  # For experiment tracking (optional)
```

##### All-in-One Installation
```bash
pip install streamlit torch torchvision ultralytics opencv-python pillow numpy matplotlib tensorflow keras scikit-learn pandas seaborn bm3d albumentations
```

**Note**: If you encounter issues with `bm3d` installation, you can skip it as it's only used for advanced denoising. The core functionality will work without it.

### Step 4: Download Pre-trained Models
The repository includes several pre-trained models:
- Models are already included in their respective directories
- For custom training, follow the notebooks in each folder

### Step 5: Verify Installation
```bash
cd app
streamlit run app.py
```
If successful, the web application will open in your browser at `http://localhost:8501`

## 🎮 Usage

### 🌐 Running the Web Application

The web application provides an intuitive interface for brain tumor detection:

```bash
cd app
streamlit run app.py
```

**Features of the Web App:**
- 📤 **Upload MRI Images**: Support for PNG, JPG, JPEG formats
- 🔍 **Real-time Analysis**: Instant tumor detection and classification  
- 📊 **Visual Results**: Bounding boxes with confidence scores
- 🎨 **Color-coded Labels**: Different colors for each tumor type
- 📋 **Detailed Reports**: Classification probabilities and model confidence

**Usage Steps:**
1. Launch the application using the command above
2. Open your browser and navigate to `http://localhost:8501`
3. Upload an MRI scan using the file uploader
4. Click "Predict" to analyze the image
5. View results with bounding boxes and classifications

### 📚 Training Your Own Models

Each directory contains comprehensive Jupyter notebooks for model training:

#### 🧠 CNN Model Training
```bash
cd "CNN+computer vision"
jupyter notebook final.ipynb
```
- **Dataset Preparation**: Instructions for organizing your MRI dataset
- **Preprocessing**: Image normalization, augmentation, and edge detection
- **Model Architecture**: Customizable CNN layers and hyperparameters
- **Training Process**: Step-by-step model training with validation
- **Evaluation**: Accuracy metrics, confusion matrices, and ROC curves

#### 🔄 Ensemble Model Training
```bash
cd "Ensemble+Computer vision"
jupyter notebook 1.ipynb
```
- **Base Models**: Multiple classifier implementations
- **Voting Strategies**: Hard and soft voting mechanisms
- **Cross-validation**: K-fold validation for robust evaluation
- **Hyperparameter Tuning**: Grid search and random search optimization

#### 🎯 U-Net Segmentation Training
```bash
cd Unet
jupyter notebook CV_FINAL.ipynb
```
- **Segmentation Masks**: Creating pixel-level tumor annotations
- **Data Augmentation**: Specialized augmentations for medical images
- **Loss Functions**: Dice coefficient, IoU, and focal loss implementations
- **Post-processing**: Morphological operations and contour refinement

#### 📱 YOLO Training (v8 and v11)
```bash
# For YOLOv8
cd YOLO-V8
jupyter notebook brain-tumour-classification.ipynb

# For YOLOv11
cd YOLO-V11
jupyter notebook mri-detection.ipynb
```
- **Dataset Annotation**: YOLO format label preparation
- **Transfer Learning**: Using pre-trained weights for medical images
- **Custom Architecture**: Modifying YOLO for medical image analysis
- **Inference Optimization**: TensorRT and ONNX conversion for deployment

### 📊 Model Evaluation and Testing

Each notebook includes comprehensive evaluation sections:

```python
# Example evaluation code
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Generate predictions
predictions = model.predict(test_data)

# Classification report
print(classification_report(y_true, y_pred, 
                          target_names=['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']))

# Confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()
```

## 🛠️ Technologies Used

### Deep Learning Frameworks
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) **TensorFlow/Keras**: CNN and U-Net implementations
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) **PyTorch**: Advanced model architectures and YOLO integration
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn**: Ensemble methods and evaluation metrics

### Computer Vision & Image Processing
- ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=opencv&logoColor=white) **OpenCV (cv2)**: Image preprocessing, edge detection, morphological operations
- **BM3D**: Advanced image denoising algorithm specifically designed for medical images
- **Albumentations**: High-performance image augmentation library
- ![PIL](https://img.shields.io/badge/Pillow-3776ab?style=flat) **Pillow (PIL)**: Basic image operations and format conversions

### Object Detection
- **YOLO Framework**: Both YOLOv8 and YOLOv11 implementations
- **Ultralytics**: Modern YOLO implementation with PyTorch backend
- **Non-Maximum Suppression**: Advanced post-processing for accurate detections

### Data Science & Visualization  
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) **NumPy**: Numerical computations and array operations
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas**: Data manipulation and analysis
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) **Matplotlib**: Statistical plotting and result visualization
- **Seaborn**: Advanced statistical data visualization

### Web Application & Deployment
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) **Streamlit**: Interactive web application framework
- **HTML/CSS**: Custom styling for enhanced user interface

### Development & Model Management
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) **Jupyter Notebooks**: Interactive development and experimentation
- **Git**: Version control and collaborative development
- **ONNX** (Optional): Model optimization and cross-platform deployment

## 📊 Dataset Information

### 🏥 Medical Image Datasets

The project is designed to work with standard medical imaging datasets:

#### **Primary Datasets**
- **Brain Tumor Classification Dataset**: Contains MRI scans with labeled tumor regions
- **Multi-class Brain Tumor Dataset**: Includes Glioma, Meningioma, and Pituitary tumor samples
- **Binary Classification Dataset**: Tumor vs. No Tumor samples for detection tasks

#### **Data Characteristics**
- **Format**: DICOM, PNG, JPEG medical images
- **Resolution**: Various resolutions (typically 256x256 to 512x512 pixels)
- **Modality**: T1-weighted, T2-weighted, and FLAIR MRI sequences
- **Annotations**: Bounding boxes and segmentation masks where applicable

#### **Data Preprocessing Pipeline**
```python
# Example preprocessing steps
def preprocess_mri(image):
    # Normalize pixel values
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply BM3D denoising
    denoised = bm3d.bm3d(image, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    
    # Resize to standard dimensions
    resized = cv2.resize(denoised, (512, 512))
    
    return resized
```

#### **Tumor Classification Categories**
- 🔴 **Glioma**: Aggressive brain tumors arising from glial cells
- 🟠 **Meningioma**: Usually benign tumors of the brain's protective membranes
- 🔵 **Pituitary Adenoma**: Tumors affecting the pituitary gland
- ⚫ **Healthy/Background**: Normal brain tissue without tumors

### 📈 Data Distribution
- **Training Set**: ~70% of total dataset
- **Validation Set**: ~15% of total dataset  
- **Test Set**: ~15% of total dataset
- **Augmented Samples**: 3-5x increase through data augmentation

### 🔒 Data Privacy and Ethics
- All datasets used are publicly available or properly anonymized
- Patient privacy is maintained according to medical data handling standards
- The system is designed as a diagnostic aid, not a replacement for medical expertise

### 📚 Recommended Datasets
For training your own models, consider these publicly available datasets:
- **Brain Tumor Classification (MRI)** - Kaggle
- **BraTS (Brain Tumor Segmentation) Challenge** - Medical Decathlon
- **TCIA (The Cancer Imaging Archive)** - NIH Database

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 🚨 Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install PyTorch with proper CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: `ImportError: No module named 'bm3d'`
```bash
# Solution: Install BM3D for image denoising
pip install bm3d

# If compilation fails, try:
conda install -c conda-forge bm3d
```

**Problem**: Streamlit app doesn't start
```bash
# Solution: Ensure Streamlit is properly installed
pip install --upgrade streamlit
streamlit --version
```

#### 🖥️ Runtime Issues

**Problem**: CUDA out of memory
```python
# Solution: Reduce batch size or use CPU
device = 'cpu'  # Force CPU usage
# Or reduce image dimensions in preprocessing
```

**Problem**: Model loading errors
```bash
# Solution: Ensure model files are in correct locations
ls app/best.pt  # Check if model file exists
# Download models if missing from respective directories
```

**Problem**: Web app slow predictions
- **GPU Acceleration**: Ensure CUDA is properly configured
- **Model Optimization**: Use TensorRT or ONNX for faster inference
- **Image Size**: Reduce input image resolution for faster processing

#### 📊 Model Performance Issues

**Problem**: Poor accuracy on custom data
- **Data Quality**: Ensure images are properly preprocessed
- **Domain Adaptation**: Fine-tune models on your specific dataset
- **Class Imbalance**: Use weighted loss functions or data balancing techniques

### 💡 Performance Optimization

```python
# Example: Optimizing model for inference
import torch.quantization

# Quantize model for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 🤝 Contributing

We welcome contributions to improve the Brain Tumor Detection System! Here's how you can help:

### 🛠️ Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Brain_Tumour_Detection.git
   cd Brain_Tumour_Detection
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes and Test**
   ```bash
   # Make your changes
   # Test the web application
   cd app && streamlit run app.py
   ```

4. **Submit Pull Request**
   ```bash
   git add .
   git commit -m "Add: Your descriptive commit message"
   git push origin feature/your-feature-name
   ```

### 📝 Contribution Guidelines

- **Code Style**: Follow PEP 8 guidelines for Python code
- **Documentation**: Update README.md and add docstrings to functions
- **Testing**: Include tests for new features
- **Medical Accuracy**: Ensure medical relevance and accuracy of implementations

### 🎯 Areas for Contribution

- 🔬 **New Model Architectures**: Implement latest research papers
- 📊 **Data Augmentation**: Advanced augmentation techniques for medical images
- 🌐 **Web Interface**: Improve UI/UX of the Streamlit application
- 📱 **Mobile App**: Develop mobile application for field usage
- 🔧 **Optimization**: Model compression and deployment optimizations
- 📚 **Documentation**: Improve tutorials and documentation

### 🐛 Bug Reports

When reporting bugs, please include:
- Python version and operating system
- Complete error traceback
- Steps to reproduce the issue
- Sample images (if applicable and privacy-compliant)

## 🚀 Future Work & Roadmap

### 🔬 Technical Enhancements

#### **Advanced Architectures**
- [ ] **Vision Transformers (ViT)**: Implement attention-based models for medical imaging
- [ ] **3D CNNs**: Extend to volumetric MRI analysis for better spatial understanding
- [ ] **Graph Neural Networks**: Model relationships between different brain regions
- [ ] **Federated Learning**: Enable collaborative training across medical institutions

#### **Multi-Modal Integration**  
- [ ] **Multi-Sequence MRI**: Combine T1, T2, FLAIR, and DWI sequences
- [ ] **Clinical Data Integration**: Incorporate patient history and demographics
- [ ] **Radiomics Features**: Extract quantitative features for enhanced analysis
- [ ] **Temporal Analysis**: Track tumor progression over time

#### **Advanced Computer Vision**
- [ ] **Self-Supervised Learning**: Reduce dependency on labeled medical data
- [ ] **Weakly Supervised Learning**: Learn from image-level labels without pixel annotations
- [ ] **Domain Adaptation**: Adapt models across different MRI scanners and protocols
- [ ] **Uncertainty Quantification**: Provide confidence measures for clinical decisions

### 🏥 Clinical Applications

#### **Expanded Pathology Detection**
- [ ] **Multiple Sclerosis**: Detect and segment MS lesions
- [ ] **Stroke Analysis**: Identify ischemic and hemorrhagic strokes  
- [ ] **Alzheimer's Disease**: Early detection of neurodegenerative changes
- [ ] **Brain Metastases**: Detect secondary tumors from other primary sites

#### **Clinical Decision Support**
- [ ] **Treatment Planning**: AI-assisted radiation therapy and surgical planning
- [ ] **Prognosis Prediction**: Estimate patient outcomes and survival rates
- [ ] **Drug Response**: Predict response to different treatment modalities
- [ ] **Risk Stratification**: Classify patients into risk categories

### 💻 Technology & Deployment

#### **Platform Development**
- [ ] **Mobile Application**: iOS and Android apps for point-of-care diagnosis
- [ ] **Web-Based Platform**: Cloud deployment for multi-institutional access
- [ ] **PACS Integration**: Direct integration with hospital imaging systems
- [ ] **API Development**: RESTful APIs for third-party integrations

#### **Performance & Scalability**
- [ ] **Edge Computing**: Deploy models on local hospital hardware
- [ ] **Real-time Processing**: Sub-second inference for emergency cases
- [ ] **Batch Processing**: Handle large-scale screening programs
- [ ] **Model Compression**: Optimize models for resource-constrained environments

### 🌐 Research & Collaboration

#### **Open Science Initiatives**
- [ ] **Public Datasets**: Contribute to open medical imaging datasets
- [ ] **Benchmark Challenges**: Participate in medical imaging competitions
- [ ] **Research Collaborations**: Partner with medical institutions and universities
- [ ] **Reproducible Research**: Provide complete experimental protocols

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary
- ✅ **Commercial Use**: Use in commercial applications
- ✅ **Modification**: Modify and adapt the code
- ✅ **Distribution**: Distribute original or modified versions
- ✅ **Private Use**: Use for personal or internal purposes
- ❗ **Limitation**: No warranty or liability from authors

**Medical Disclaimer**: This software is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified healthcare professionals for medical decisions.

## 🙏 Acknowledgments

### 🏛️ **Research & Academic Contributions**
- **Medical Imaging Community**: For advancing the field of computer-aided diagnosis
- **Open Source Contributors**: Developers of TensorFlow, PyTorch, and OpenCV
- **YOLO Developers**: Ultralytics team for modern object detection frameworks
- **Medical Institutions**: Providing anonymized datasets for research

### 📚 **Key References**
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- YOLOv8: Real-Time Object Detection and Image Segmentation
- BM3D: Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering
- Deep Learning for Medical Image Analysis: A Comprehensive Review

### 🤝 **Special Thanks**
- **Group 27 Team**: Original developers and researchers
- **Medical Professionals**: For validation and clinical insights  
- **Beta Testers**: Early adopters who provided valuable feedback
- **Open Source Community**: For continuous support and contributions

---

## 📞 Support & Contact

### 💬 **Getting Help**
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/hiteshshanmukha/Brain_Tumour_Detection/issues)
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check this README and inline code documentation

### 🏥 **For Medical Institutions**
If you're a medical institution interested in collaboration or clinical validation:
- Reach out through GitHub issues with the "clinical-collaboration" label
- Ensure compliance with local medical data regulations (HIPAA, GDPR, etc.)
- Consider institutional review board (IRB) approval for research use

### 🎓 **For Researchers & Students**
- Fork the repository for your research projects
- Cite this work in your publications if it contributes to your research
- Contribute improvements back to the community

---

<div align="center">

**⚕️ Advancing Medical AI for Better Healthcare ⚕️**

*Built with ❤️ for the medical community*

[![GitHub stars](https://img.shields.io/github/stars/hiteshshanmukha/Brain_Tumour_Detection?style=social)](https://github.com/hiteshshanmukha/Brain_Tumour_Detection/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hiteshshanmukha/Brain_Tumour_Detection?style=social)](https://github.com/hiteshshanmukha/Brain_Tumour_Detection/network/members)

</div>


