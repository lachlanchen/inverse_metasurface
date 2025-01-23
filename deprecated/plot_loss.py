import re
import matplotlib.pyplot as plt

def parse_loss_file(filename):
    """
    Parses each line of the loss file and extracts epoch, train loss, and test loss.
    """
    epochs = []
    train_losses = []
    test_losses = []
    pattern = r"\[Epoch (\d+)/(\d+)\]\s*TrainLoss=(\d+\.\d+)\s*TestLoss=(\d+\.\d+)"
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(3))
                test_loss = float(match.group(4))
                epochs.append(epoch)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
    return epochs, train_losses, test_losses

def main():
    """
    Reads the supervised_loss.txt file, parses the data, and plots the losses.
    """
    epochs, train_losses, test_losses = parse_loss_file("supervised_loss.txt")
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Over Epochs")
    plt.legend()
    plt.savefig("loss_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

