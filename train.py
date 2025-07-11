"""
YOLOv8 Training Module for Zucchini Leaf Instance Segmentation
Based on superior performance demonstrated in Vineyard Vision research
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from config import get_config
from dataset_handler import COCODatasetHandler

class ZucchiniLeafTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        """Setup training device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f" Using CUDA GPU: {torch.cuda.get_device_name()}")
                print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = "cpu"
                print("Using CPU")
        else:
            device = self.config.device
        
        return device
    
    def prepare_dataset(self):
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        # Initialize dataset handler
        dataset_handler = COCODatasetHandler(self.config)
        dataset_handler.discover_datasets()
        
        # Validate datasets
        stats = dataset_handler.validate_datasets()
        
        # Convert to YOLO format
        yolo_dataset_path = Path(self.config.output_dir) / "dataset_yolo"
        dataset_handler.create_yolo_format(str(yolo_dataset_path))
        
        return str(yolo_dataset_path / "data.yaml")
    
    def load_model(self):
        """Load YOLOv8 model"""
        print(f"Loading YOLOv8 model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        return self.model
    
    def train(self, data_yaml_path, resume=False):
        """Execute training pipeline based on Vineyard Vision methodology"""
        print("Starting YOLOv8 training...")
        
        if self.model is None:
            self.load_model()
        
        # Training arguments based on research findings
        train_args = {
            'data': data_yaml_path,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            'device': self.device,
            'project': self.config.output_dir,
            'name': self.config.experiment_name,
            'save': True,
            'val': True,
            'plots': True,
            'verbose': True,
            'patience': self.config.patience,
            'lr0': self.config.learning_rate,
            'amp': self.config.amp,
            'workers': self.config.num_workers,
            'resume': resume
        }
        
        print("⚙️ Training configuration:")
        for key, value in train_args.items():
            print(f"   - {key}: {value}")
        
        # Start training
        start_time = time.time()
        
        try:
            results = self.model.train(**train_args)
            training_time = time.time() - start_time
            
            print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
            print(f"Results: {results}")
            
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise e

def main():
    """Example training execution"""
    config = get_config()
    trainer = ZucchiniLeafTrainer(config)
    
    # Prepare dataset
    data_yaml_path = trainer.prepare_dataset()
    
    # Train model
    results = trainer.train(data_yaml_path)
    
    print("✅ Training pipeline completed!")

if __name__ == "__main__":
    main()
