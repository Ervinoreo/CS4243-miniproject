import os
import sys
import json
import matplotlib.pyplot as plt

def plot_training_history(folder_name):
    json_path = os.path.join(folder_name, "training_history.json")
    if not os.path.exists(json_path):
        print(f"No training_history.json found in {folder_name}")
        return

    # Load JSON
    with open(json_path, "r") as f:
        history = json.load(f)

    # Check for correct keys
    required_keys = ["train_loss", "train_acc", "val_loss", "val_acc"]
    for key in required_keys:
        if key not in history:
            print(f"Key '{key}' not found in JSON.")
            return

    # Prepare data
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = range(1, len(train_loss) + 1)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Set the main title as folder name
    plt.suptitle(folder_name)

    # Save the plot
    output_path = os.path.join(folder_name, "training_history.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_history.py <folder_name>")
    else:
        folder_name = sys.argv[1]
        plot_training_history(folder_name)
