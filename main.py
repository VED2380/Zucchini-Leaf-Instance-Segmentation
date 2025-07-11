"""
Main Execution Script for Zucchini Leaf Instance Segmentation
Complete pipeline based on Vineyard Vision research
"""

import argparse
import sys
from pathlib import Path
from config import get_config
from dataset_handler import COCODatasetHandler
from train import ZucchiniLeafTrainer
from evaluate import ZucchiniLeafEvaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Zucchini Leaf Instance Segmentation Pipeline - Based on Vineyard Vision Research"
    )
    
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["prepare", "train", "evaluate", "predict", "all"],
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--dataset-root", type=str, default="./dataset",
        help="Path to dataset root directory"
    )
    
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model for evaluation/prediction"
    )
    
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory containing images for prediction"
    )
    
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint"
    )
    
    return parser.parse_args()

def main():
    """Main execution pipeline"""
    print("🌿 Zucchini Leaf Instance Segmentation Pipeline")
    print("   Based on Vineyard Vision Research (YOLOv8: 88.5% mAP)")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config()
    config.dataset_root = args.dataset_root
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    
    # Track important paths
    data_yaml_path = Path(config.output_dir) / "dataset_yolo" / "data.yaml"
    best_model_path = Path(config.output_dir) / config.experiment_name / "weights" / "best.pt"
    
    # Initialize components
    dataset_handler = None
    trainer = None
    evaluator = None
    
    try:
        # Stage 1: Dataset Preparation
        if args.stage in ["prepare", "all"]:
            print("\nSTAGE 1: DATASET PREPARATION")
            print("-" * 40)
            
            dataset_handler = COCODatasetHandler(config)
            dataset_handler.discover_datasets()
            stats = dataset_handler.validate_datasets()
            
            dataset_handler.create_yolo_format(str(Path(config.output_dir) / "dataset_yolo"))
            
            print(f"Dataset preparation completed")
            print(f"   - Total images: {stats['total_images']}")
            print(f"   - Total annotations: {stats['total_annotations']}")
        
        # Stage 2: Model Training
        if args.stage in ["train", "all"]:
            print("\nSTAGE 2: MODEL TRAINING")
            print("-" * 40)
            
            trainer = ZucchiniLeafTrainer(config)
            
            # Prepare dataset if not done already
            if not data_yaml_path.exists() or args.stage == "all":
                if dataset_handler is None:
                    dataset_handler = COCODatasetHandler(config)
                    dataset_handler.discover_datasets()
                
                dataset_handler.create_yolo_format(str(Path(config.output_dir) / "dataset_yolo"))
            
            # Train model
            trainer.load_model()
            results = trainer.train(str(data_yaml_path), resume=args.resume)
            
            print("Training completed")
        
        # Stage 3: Model Evaluation
        if args.stage in ["evaluate", "all"] and (args.model_path or best_model_path.exists()):
            print("\nSTAGE 3: MODEL EVALUATION")
            print("-" * 40)
            
            model_path = args.model_path if args.model_path else str(best_model_path)
            
            evaluator = ZucchiniLeafEvaluator(config)
            evaluator.load_model(model_path)
            
            # Evaluate on dataset
            if data_yaml_path.exists():
                metrics = evaluator.evaluate_model(str(data_yaml_path))
                print("Evaluation completed")
            else:
                print("No dataset configuration found. Skipping evaluation.")
        
        # Stage 4: Prediction on new images
        if args.stage in ["predict"] and args.image_dir:
            print("\nSTAGE 4: PREDICTION")
            print("-" * 40)
            
            if not (args.model_path or best_model_path.exists()):
                print("No trained model found. Please train a model first.")
                sys.exit(1)
            
            model_path = args.model_path if args.model_path else str(best_model_path)
            
            if evaluator is None:
                evaluator = ZucchiniLeafEvaluator(config)
                evaluator.load_model(model_path)
            
            # Generate predictions
            output_dir = Path(config.output_dir) / "predictions"
            predictions = evaluator.generate_predictions(args.image_dir, str(output_dir))
            
            print(f"Generated predictions for {len(predictions)} images")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
    
    print("\nPipeline execution completed successfully!")
    print("\nNext steps:")
    print("1. Check results in the outputs/ directory")
    print("2. Review training metrics and plots")
    print("3. Use the trained model for prediction on new images")

if __name__ == "__main__":
    main()
