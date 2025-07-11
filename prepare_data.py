# prepare_data.py (Final Version for WSL)

import os
import json
import shutil
import glob
from tqdm import tqdm
from pycocotools.coco import COCO

def find_image_recursively(root_dir, image_name):
    """This function searches for an image file within a directory and its subdirectories."""
    if not os.path.isdir(root_dir):
        return None
    image_subdir_path = os.path.join(root_dir, 'images', image_name)
    if os.path.exists(image_subdir_path):
        return image_subdir_path
    for dirpath, _, filenames in os.walk(root_dir):
        if image_name in filenames:
            return os.path.join(dirpath, image_name)
    return None

def convert_coco_to_yolo(coco_json_path, output_dir, original_image_dir):
    """Converts a single COCO dataset to YOLOv8 format."""
    print(f"Processing {coco_json_path}...")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    coco = COCO(coco_json_path)

    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for img_info in tqdm(coco_data['images'], desc=f"Converting {os.path.basename(original_image_dir)}"):
        image_id = img_info['id']
        image_name = img_info['file_name']
        image_width = img_info['width']
        image_height = img_info['height']

        original_img_path = find_image_recursively(original_image_dir, image_name)
        if not original_img_path:
            print(f"Warning: Image '{image_name}' not found in '{original_image_dir}' or its subdirectories. Skipping.")
            continue

        shutil.copy(original_img_path, os.path.join(images_dir, image_name))

        label_file_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
        
        with open(label_file_path, 'w') as label_file:
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                category_id = 0
                segmentation = ann['segmentation']
                if not segmentation: continue
                for seg_poly in segmentation:
                    normalized_points = []
                    for i in range(0, len(seg_poly), 2):
                        x = seg_poly[i] / image_width
                        y = seg_poly[i+1] / image_height
                        normalized_points.extend([str(x), str(y)])
                    label_file.write(f"{category_id} {' '.join(normalized_points)}\n")

if __name__ == '__main__':
    # --- FINAL PATH CORRECTION FOR WSL (LINUX) ENVIRONMENT ---
    # The Windows path C:\... has been converted to the WSL path /mnt/c/...
    base_dataset_path = '/mnt/c/code/iit_project/Dataset'
    output_base_path = './yolov8_dataset'

    # Automatically find all 'annotations.coco.json' files recursively
    search_pattern = os.path.join(base_dataset_path, '**', 'annotations.coco.json')
    annotation_files = glob.glob(search_pattern, recursive=True)

    if not annotation_files:
        print(f"Error: No 'annotations.coco.json' files found inside '{base_dataset_path}'. Please check the path.")
    else:
        print(f"Found {len(annotation_files)} annotation files to process.")

    # Loop through each found annotation file and process it
    for coco_json_path in annotation_files:
        original_image_dir = os.path.dirname(coco_json_path)
        relative_path = os.path.relpath(original_image_dir, base_dataset_path)
        output_dir = os.path.join(output_base_path, relative_path)
        convert_coco_to_yolo(coco_json_path, output_dir, original_image_dir)

    print("\nCOCO to YOLOv8 conversion complete for all specified datasets.")
    print("Creating zucchini_leaf_data.yaml...")
    
    yolo_image_dirs = glob.glob(os.path.join(output_base_path, '**', 'images'), recursive=True)
    
    train_paths, val_paths, test_paths = [], [], []

    for img_dir in yolo_image_dirs:
        path_lower = img_dir.lower()
        relative_path = os.path.relpath(img_dir, output_base_path).replace(os.sep, '/')
        
        if 'train' in path_lower:
            train_paths.append(relative_path)
        elif 'valid' in path_lower:
            val_paths.append(relative_path)
        elif 'test' in path_lower:
            test_paths.append(relative_path)

    abs_output_path = os.path.abspath(output_base_path).replace(os.sep, '/')
    
    data_yaml_content = f"""
path: {abs_output_path}
train: {train_paths}
val: {val_paths}
test: {test_paths}

nc: 1
names: ['zucchini_leaf']
"""

    with open("zucchini_leaf_data.yaml", "w") as f:
        f.write(data_yaml_content)
    print("\nzucchini_leaf_data.yaml updated successfully.")