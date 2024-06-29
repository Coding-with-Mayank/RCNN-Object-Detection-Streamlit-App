import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision.transforms as T


class CustomCSVDetectionDataset(Dataset):
    def __init__(self, csv_file, images_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_ids = self.df['image_name'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['image_name'] == image_id]

        image = Image.open(os.path.join(self.images_dir, image_id)).convert("RGB")
        boxes = []
        labels = []

        for _, row in records.iterrows():
            xmin = row['bbox_x']
            ymin = row['bbox_y']
            xmax = xmin + row['bbox_width']
            ymax = ymin + row['bbox_height']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1 if row['label_name'] == 'Marine Animal' else 2)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        # Convert image to tensor if no transformations are applied
        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        return image, target

# Define paths
csv_file = 'Dataset7.0/train/annotations/train_labels.csv'
images_dir = 'content/drive/MyDrive/Dataset7.0/train/images'

# Create dataset and dataloaders without transformations
dataset = CustomCSVDetectionDataset(csv_file, images_dir, transforms=None)

# DataLoader setup
def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize the model
model = get_model(num_classes=3)  # 2 classes (marine animal, trash) + background
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
import torch.optim as optim

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define the path to your model weights
model_path = '/home/navarch/train_rcnn/model_weights_epoch_53.pth'

# Initialize the model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Create an instance of the model
model = get_model(num_classes=3)  # Replace 3 with the number of classes in your dataset

# Load the model weights onto CPU
device = torch.device('cpu')  # Specify the device (CPU in this case)
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
model.eval()

# Print a message indicating successful loading
print(f"Model loaded from {model_path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_colors = {
    1: (255, 165, 0),  # Marine Animal - Orange
    2: (255, 255, 0)   # Trash - Yellow
}

class_names = {
    1: 'Marine Animal',
    2: 'Trash'
}

label_to_int = {
    'Marine Animal': 1,
    'Trash': 2
}

def visualize_predictions(image, boxes, labels, scores, threshold=0.5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            box = box.astype(np.int32)
            color = class_colors.get(label, (255, 255, 255))
            label_name = class_names.get(label, 'Unknown')

            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image, f"{label_name}: {score:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

app = Flask(__name__)

# Define the path to your model weights
model_path = '/home/navarch/train_rcnn/model_weights_epoch_53.pth'

# Function to load the model
def load_model(model_path):
    # Define your model architecture here
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 3  # Replace with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the model weights onto CPU
    device = torch.device('cpu')  # Use CPU for deployment
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model

# Load the model
model = load_model(model_path)

# Function to preprocess and predict
def predict_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = ToTensor()(image).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform prediction
    predictions = predict_image(image, model)

    # Process predictions and format output (similar to previous implementation)

    # Example: Extract bounding boxes, labels, and scores from predictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # Visualize predictions on the image
    result_image = visualize_predictions(image, pred_boxes, pred_labels, pred_scores)

    # Convert result image to base64 to send to frontend
    _, img_encoded = cv2.imencode('.jpg', result_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
