from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pickle
import time
from argparse import ArgumentParser

from coding_hw.coding_hw4 import train_nnet, train_nnet_np, evaluate_nnet_np


def evaluate_nnet(nnet: nn.Module, data_input_np, data_labels_np):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_input = torch.tensor(data_input_np).float()
    val_labels = torch.tensor(data_labels_np).long()
    nnet_output: Tensor = nnet(val_input).detach()

    loss = criterion(nnet_output, val_labels)

    nnet_label = np.argmax(nnet_output.data.numpy(), axis=1)
    acc: float = 100 * np.mean(nnet_label == val_labels.data.numpy())

    return loss.item(), acc


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--extra', default=False, action='store_true', help="")
    args = parser.parse_args()

    # parse data
    train_input_np, train_labels_np = pickle.load(open("data/mnist/mnist_train.pkl", "rb"))
    train_input_np = train_input_np.reshape(-1, 28 * 28)

    val_input_np, val_labels_np = pickle.load(open("data/mnist/mnist_val.pkl", "rb"))
    val_input_np = val_input_np.reshape(-1, 28 * 28)

    print(f"Training input shape: {train_input_np.shape}, Validation data shape: {val_input_np.shape}")

    # get nnet
    start_time = time.time()
    if not args.extra:
        nnet: nn.Module = train_nnet(train_input_np, train_labels_np, val_input_np, val_labels_np)
        loss, acc = evaluate_nnet(nnet, val_input_np, val_labels_np)
    else:
        nnet: Callable = train_nnet_np(train_input_np, train_labels_np, val_input_np, val_labels_np)
        loss, acc = evaluate_nnet_np(nnet, val_input_np, val_labels_np)
    print(f"Loss: %.5f, Accuracy: %.2f%%, Time: %.2f seconds" % (loss, acc, time.time() - start_time))


if __name__ == "__main__":
    main()
