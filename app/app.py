import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from collections import OrderedDict
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import yaml
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataclasses import dataclass
from matplotlib.colors import to_rgba
from collections import OrderedDict
# Define class labels and colors
import os
os.environ['ULTRALYTICS_HUB'] = '0'  # disables hub interaction

from ultralytics import YOLO

class_labels = { 
    0: 'Glioma',
    1: 'Meningioma',
    3: 'Pituitary', 
    2: 'Background', # Reorder to have background last, but keep mapping
}

class_colors = {
    0: 'darkmagenta',
    1: 'orange',
    3: 'darkcyan',
    2: 'darkslategray',
}
confidence_threshold = 0.5

# Plotting functions
def plot_boxes(ax, image, boxes, labels, prediction_confidences=None):
    ax.imshow(image)
    for i, label in enumerate(labels):
        x_min, y_min, x_max, y_max = boxes[i]
        width, height = x_max - x_min,  y_max - y_min
        class_label = class_labels[int(label)]
        class_label += ' tumor' if class_label != 'Background' else ''
        class_color = class_colors[int(label)]
        legend = class_label
        transparency = 0.4
        line_width = 2
        bounding_box = plt.Rectangle((x_min, y_min), width, height,
                                     linewidth=line_width, label=legend)
        label_text = ax.text(x_min+4, y_min-5, class_label)
        label_text.set_bbox(dict(facecolor=to_rgba(class_color, alpha=transparency)))
        if prediction_confidences is not None:
            confidence_label = f' confidence: {prediction_confidences[i]:.2f}'
            bounding_box.set_edgecolor(class_color)
            bounding_box.set_facecolor('none')
            bounding_box.set_label('Predicted ' + legend)
            label_text.set_text(class_label + confidence_label)
            label_box = label_text.get_bbox_patch()
            label_box.set_edgecolor(class_color)
            label_box.set_facecolor('none')
            label_box.set_linewidth(line_width)  
        else:
            bounding_box.set_label('Ground truth ' + legend)
            bounding_box.set_edgecolor('none')
            bounding_box.set_facecolor(to_rgba(class_color, alpha=transparency))
            label_box = label_text.get_bbox_patch()
            label_box.set_edgecolor('none')
        ax.add_patch(bounding_box)

def add_legends(fig):
    box_legends = [ax.get_legend_handles_labels() for ax in fig.axes]
    box_legends, labels = [sum(lol, []) for lol in zip(*box_legends)]
    if box_legends:
        box_legends = sorted(zip(labels, box_legends), key=lambda x: x[0])
        labels, box_legends = zip(*box_legends)
        box_labeled_legends = OrderedDict(zip(labels, box_legends))
        fig.legend(box_labeled_legends.values(), box_labeled_legends.keys())

def visualize_prediction(image_np, predictions):
    fig, ax = plt.subplots(figsize=(10, 10))
    boxes = predictions.boxes.xyxy.cpu().numpy()
    confidences = predictions.boxes.conf.cpu().numpy()
    classes = predictions.boxes.cls.cpu().numpy()
    mask = confidences > confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    classes = classes[mask]
    plot_boxes(ax, image_np, boxes, classes, confidences)
    add_legends(fig)
    st.pyplot(fig)

# Load best model
best_model_path = "best.pt"  # Change if needed
model = YOLO(best_model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Streamlit interface
st.title("Tumor Detection from Image")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        results = model.predict(image_pil, device=device, verbose=False)[0]
        visualize_prediction(image_np, results)
