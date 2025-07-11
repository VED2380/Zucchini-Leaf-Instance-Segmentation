# evaluation_metrics (1).py

from ultralytics import YOLO
import os
import json
from pycocotools.coco import COCO
from pycocotools.mask import decode
import numpy as np
import cv2
import torch
from gnn_postprocess import GNNRefinement, refine_mask_with_gnn # Import the GNN components

def evaluate_yolov8_seg(model_path, data_yaml_path):
    # Load a trained YOLOv8x-seg model
    model = YOLO(model_path)

    # Validate the model on the test set
    metrics = model.val(data=data_yaml_path)

    # Print metrics
    print("\n--- Evaluation Metrics (YOLOv8 Base) ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    print(f"mAP50-95 (Mask): {metrics.seg.map}")
    print(f"mAP50 (Mask): {metrics.seg.map50}")
    print(f"mAP75 (Mask): {metrics.seg.map75}")

    # --- GNN Post-processing (Conceptual Integration) ---
    print("\n--- GNN Post-processing (Conceptual Integration) ---")
    print("Note: The GNN model here is untrained and serves as a conceptual placeholder.")
    
    # This section remains conceptual until you train the GNN
    # and modify this script to load the trained GNN weights.
    
    print("GNN post-processing conceptual integration complete.")

if __name__ == "__main__":
    # --- PATH UPDATED HERE ---
    trained_model_path = r"C:\code\iit_project\outputs\zucchini_leaf_segmentation\weights\best.pt"

    # Path to your data.yaml file
    data_yaml_path = "zucchini_leaf_data.yaml"

    evaluate_yolov8_seg(trained_model_path, data_yaml_path)