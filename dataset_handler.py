"""
COCO Dataset Handler for Zucchini Leaf Instance Segmentation
Handles recursive discovery of datasets in arbitrary substructure
"""
import os
import json
import shutil
import yaml
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
from config import get_config

class COCODatasetHandler:
    def __init__(self, config):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.coco_datasets = {}
        self.image_paths = {}

    def discover_datasets(self):
        """Recursively discover all COCO datasets in the folder structure"""
        print("Recursively discovering datasets...")

        for annotation_file in self.dataset_root.rglob("annotations.coco.json"):
            try:
                # Expect structure: root/time_period/capture_mode/split/annotations.coco.json
                parts = annotation_file.parts[-4:-1]  # last 3 dirs before the file
                if len(parts) != 3:
                    print(f"Skipping invalid dataset path: {annotation_file}")
                    continue

                time_period, capture_mode, split = parts
                dataset_key = f"{time_period}/{capture_mode}/{split}"
                coco = COCO(str(annotation_file))

                self.coco_datasets[dataset_key] = coco
                self.image_paths[dataset_key] = annotation_file.parent / "images"

                print(f"Loaded: {dataset_key} - {len(coco.imgs)} images, {len(coco.anns)} annotations")

            except Exception as e:
                print(f"Failed to load {annotation_file}: {e}")

    def validate_datasets(self):
        """Validate all discovered datasets"""
        print("Validating datasets...")

        total_images = 0
        total_annotations = 0
        missing_images = []

        for dataset_key, coco in self.coco_datasets.items():
            image_path = self.image_paths[dataset_key]

            dataset_images = len(coco.imgs)
            dataset_annotations = len(coco.anns)

            total_images += dataset_images
            total_annotations += dataset_annotations

            # Check for missing image files
            for img_id, img_info in coco.imgs.items():
                img_file_path = image_path / img_info['file_name']
                if not img_file_path.exists():
                    missing_images.append(f"{dataset_key}/{img_info['file_name']}")

            print(f"{dataset_key}: {dataset_images} images, {dataset_annotations} annotations")

        print(f"Total: {total_images} images, {total_annotations} annotations")
        if missing_images:
            print(f"Missing {len(missing_images)} image files")

        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "missing_images": missing_images
        }

    def polygon_to_yolo(self, segmentation, img_width, img_height):
        """Convert COCO polygon segmentation to YOLO format"""
        if len(segmentation) < 6:  # Need at least 3 points
            return []

        normalized_points = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / img_width
            y = segmentation[i + 1] / img_height
            normalized_points.extend([x, y])

        return normalized_points

    def create_yolo_format(self, output_dir):
        """Convert all COCO datasets to YOLO format"""
        print("Converting COCO to YOLO format...")

        output_path = Path(output_dir)

        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Process each dataset
        for dataset_key, coco in tqdm(self.coco_datasets.items(), desc="Processing datasets"):
            time_period, capture_mode, split = dataset_key.split('/')

            # Map 'valid' to 'val' for YOLO format
            yolo_split = 'val' if split == 'valid' else split

            # Get image path
            image_path = self.image_paths[dataset_key]

            # Process each image
            for img_id in tqdm(coco.imgs.keys(), desc=f"Processing {dataset_key}", leave=False):
                img_info = coco.imgs[img_id]
                img_filename = img_info['file_name']

                # Create unique filename with temporal and capture info
                dest_filename = f"{time_period}_{capture_mode}_{img_filename}"
                img_path = image_path / img_filename

                if not img_path.exists():
                    continue

                # Copy image to YOLO structure
                dst_img_path = output_path / yolo_split / 'images' / dest_filename
                shutil.copy2(img_path, dst_img_path)

                # Create corresponding label file
                label_filename = Path(dest_filename).stem + '.txt'
                label_path = output_path / yolo_split / 'labels' / label_filename

                # Get annotations for this image
                ann_ids = coco.getAnnIds(imgIds=[img_id])
                annotations = coco.loadAnns(ann_ids)

                # Write YOLO format labels
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        if 'segmentation' in ann and ann['segmentation']:
                            if isinstance(ann['segmentation'], list):
                                for seg in ann['segmentation']:
                                    yolo_points = self.polygon_to_yolo(
                                        seg, img_info['width'], img_info['height']
                                    )
                                    if yolo_points:
                                        line = f"0 " + " ".join(map(str, yolo_points)) + "\n"
                                        f.write(line)

        # Create data.yaml configuration file
        data_yaml = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'leaf'},
            'nc': 1
        }

        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"YOLO format dataset created at: {output_path}")
        print(f"Configuration saved as: {output_path / 'data.yaml'}")

def main():
    """Run full dataset preparation"""
    config = get_config()
    handler = COCODatasetHandler(config)
    handler.discover_datasets()
    handler.validate_datasets()
    handler.create_yolo_format("./outputs/dataset_yolo")

if __name__ == "__main__":
    main()
