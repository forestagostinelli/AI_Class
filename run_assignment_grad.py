import torch
import torch.nn as nn
from torch import Tensor
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

from coding_hw.coding_hw_grad import train_autoencoder


def evaluate_nnet(encoder: nn.Module, decoder: nn.Module, data_np):
    encoder.eval()
    decoder.eval()
    criterion = nn.MSELoss()

    data = torch.tensor(data_np).float()
    nnet_output: Tensor = decoder(encoder(data))

    loss = criterion(nnet_output, data).detach()

    return loss.item()


def visualize_result(encoder, data_np, labels_np):
    encoder.eval()
    data: Tensor = torch.tensor(data_np).float()
    encoder_out_np = encoder(data).cpu().data.numpy()

    for label in list(np.unique(labels_np)):
        labels_mask = labels_np == label
        plt.scatter(encoder_out_np[labels_mask, 0], encoder_out_np[labels_mask, 1], label=label)

    plt.legend()
    plt.show()


def main():
    # parse data
    train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train.pkl", "rb"))
    train_input_np = train_input_np.reshape(-1, 28 * 28)

    print(f"Training input shape: {train_input_np.shape}")

    # get nnet
    start_time = time.time()
    encoder, decoder = train_autoencoder(train_input_np)
    loss: float = evaluate_nnet(encoder, decoder, train_input_np)

    print(f"Loss: %.5f, Time: %.2f seconds" % (loss, time.time() - start_time))

    visualize_result(encoder, train_input_np, train_labels_np)


if __name__ == "__main__":
    main()
