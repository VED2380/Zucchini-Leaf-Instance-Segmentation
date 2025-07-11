# Visualize YOLO-World training metrics from CSV results

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def load_results(csv_path):
    return pd.read_csv(csv_path)
import matplotlib.pyplot as plt
import os

def plot_metrics(df, save_dir):
    epochs = df['epoch']
    os.makedirs(save_dir, exist_ok=True)

    # 1. Precision and Recall
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['metrics/precision(B)'], label='Precision', marker='o')
    plt.plot(epochs, df['metrics/recall(B)'], label='Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Precision & Recall')
    plt.grid(True)
    plt.legend()
    precision_recall_path = os.path.join(save_dir, "precision_recall.png")
    plt.savefig(precision_recall_path)
    print(f" Precision & Recall plot saved to {precision_recall_path}")
    plt.close()

    # 2. mAP50 and mAP50-95
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', marker='o')
    plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Metrics')
    plt.grid(True)
    plt.legend()
    map_path = os.path.join(save_dir, "mAP_metrics.png")
    plt.savefig(map_path)
    print(f" mAP metrics plot saved to {map_path}")
    plt.close()

    # 3. Training Losses
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['train/box_loss'], label='Box Loss (train)', marker='o')
    plt.plot(epochs, df['train/cls_loss'], label='Cls Loss (train)', marker='o')
    plt.plot(epochs, df['train/dfl_loss'], label='DFL Loss (train)', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True)
    plt.legend()
    train_loss_path = os.path.join(save_dir, "training_losses.png")
    plt.savefig(train_loss_path)
    print(f" Training losses plot saved to {train_loss_path}")
    plt.close()

    # 4. Validation Losses
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['val/box_loss'], label='Box Loss (val)', marker='o')
    plt.plot(epochs, df['val/cls_loss'], label='Cls Loss (val)', marker='o')
    plt.plot(epochs, df['val/dfl_loss'], label='DFL Loss (val)', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.grid(True)
    plt.legend()
    val_loss_path = os.path.join(save_dir, "validation_losses.png")
    plt.savefig(val_loss_path)
    print(f" Validation losses plot saved to {val_loss_path}")
    plt.close()


def plot_lr(df, save_dir):
    epochs = df['epoch']
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, df['lr/pg0'], label='LR (param group 0)', marker='o')
    plt.plot(epochs, df['lr/pg1'], label='LR (param group 1)', marker='o')
    plt.plot(epochs, df['lr/pg2'], label='LR (param group 2)', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    lr_path = os.path.join(save_dir, "learning_rate.png")
    plt.savefig(lr_path)
    print(f" Learning rate plot saved to {lr_path}")
    plt.show()

def main(args):
    df = load_results(args.csv)
    plot_metrics(df, args.save_dir)
    plot_lr(df, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize YOLO-World training metrics.')
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the results CSV file (e.g., results.csv)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./plots',
        help='Directory to save generated plots'
    )
    args = parser.parse_args()

    main(args)
