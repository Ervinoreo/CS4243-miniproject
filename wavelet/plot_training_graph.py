import json
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_history(history_file, output_dir=None):
    """Plot training history from JSON file"""
    
    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add best validation accuracy annotation
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    ax2.annotate(f'Best: {best_val_acc:.2f}% (Epoch {best_epoch})', 
                xy=(best_epoch, best_val_acc), 
                xytext=(best_epoch + len(epochs)*0.1, best_val_acc - 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(history_file)
    
    plot_path = os.path.join(output_dir, 'training_history_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    
    # Display summary statistics
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs: {len(epochs)}")
    print(f"   Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"   Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Check for overfitting
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    if acc_gap > 10:
        print(f"⚠️  Warning: Possible overfitting detected (train-val accuracy gap: {acc_gap:.2f}%)")
    elif acc_gap > 5:
        print(f"⚠️  Moderate overfitting detected (train-val accuracy gap: {acc_gap:.2f}%)")
    else:
        print(f"✅ Good generalization (train-val accuracy gap: {acc_gap:.2f}%)")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training history from JSON file")
    parser.add_argument("history_file", type=str, help="Path to training_history.json file")
    parser.add_argument("--output_dir", type=str, help="Directory to save the plot (default: same as history file)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_file):
        print(f"Error: File {args.history_file} does not exist")
        return
    
    plot_training_history(args.history_file, args.output_dir)


if __name__ == "__main__":
    main()