import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curve(csv_path, save_dir, arch_name):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Train_mIoU'], label='Train mIoU', color='blue', linewidth=2)
    plt.plot(df['Epoch'], df['Val_mIoU'], label='Val mIoU', color='orange', linewidth=2)
    
    # Dynamically find and plot the best validation peak
    best_val = df['Val_mIoU'].max()
    best_epoch = df.loc[df['Val_mIoU'].idxmax(), 'Epoch']
    
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Peak ({best_val:.4f})')
    plt.title(f'{arch_name} - Training and Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{arch_name}_training_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the training_log.csv")
    parser.add_argument("--out", type=str, required=True, help="Directory to save the plot")
    parser.add_argument("--name", type=str, required=True, help="Name of the architecture for the title")
    
    args = parser.parse_args()
    plot_training_curve(args.csv, args.out, args.name)