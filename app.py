import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from rcnn_deploy import load_model
import base64
import io

# Import your R-CNN model definition and loading code
import rcnn_deploy

# Define class colors and names (modify based on your dataset)
class_colors = {
    1: (255, 165, 0),  # Class 1 color (e.g., blue)
    2: (255, 255, 0)   # Class 2 color (e.g., yellow)
}

class_names = {
    1: 'Class 1 Name',  # Replace with actual class names
    2: 'Class 2 Name'
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


def main():

    st.title("Object Detection App")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Define the path to your model weights
        model_path = '/home/navarch/train_rcnn/model_weights_epoch_53.pth'

        # loading the model 
        model = load_model(model_path)

        predictions = rcnn_deploy.predict_image(image_array,model)

        if predictions is None:  # Handle potential errors from rcnn_deploy.py
            st.error("Error occurred during detection. Check model and image format.")
            return

        # Extract bounding boxes, labels, and scores
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()

        # Visualize predictions on the image
        result_image = visualize_predictions(image_array, pred_boxes, pred_labels, pred_scores)
        
        # Saving processed image 
        cv2.imwrite("processed_image.jpg", result_image)
        # Convert result image to base64 for display in Streamlit
        _, img_encoded = cv2.imencode('.jpg', result_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        print(f"Encoded Image Data (First 100 characters): {img_base64[:100]}...")
        img_data = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_data))
        st.subheader("Detected Objects")
        st.image(image, width=image.width, use_column_width=True)

        # Optionally display additional information like detected class names and scores
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.5:  # Adjust threshold as needed
                st.write(f"- {class_names[label]} (Confidence: {score:.2f})")


if __name__ == '__main__':
    main()