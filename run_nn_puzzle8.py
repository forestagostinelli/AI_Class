from proj_code.proj3 import get_nnet_model, train_nnet
import math
from typing import List
import numpy as np

import torch
from torch import nn

import pickle


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def sample_training_data(states_nnet: np.ndarray, outputs: np.ndarray, num_samp_total: int):
    max_cost_to_go: int = int(np.max(outputs))

    samp_idxs: np.array = np.zeros(0, dtype=np.int)
    num_per_cost_to_go: List[int] = split_evenly(num_samp_total, max_cost_to_go + 1)

    for cost_to_go, num_samp in zip(range(max_cost_to_go + 1), num_per_cost_to_go):
        ctg_idxs = np.where(outputs[:, 0] == cost_to_go)[0]
        ctg_samp_idxs = np.random.choice(ctg_idxs, size=num_samp)

        samp_idxs = np.concatenate((samp_idxs, ctg_samp_idxs))

    np.random.shuffle(samp_idxs)

    states_nnet_samp = states_nnet[samp_idxs].astype(np.float32)
    outputs_samp: np.ndarray = outputs[samp_idxs].astype(np.float32)

    return states_nnet_samp, outputs_samp


def main():
    torch.set_num_threads(1)

    # get nnet model
    nnet: nn.Module = get_nnet_model()
    device = torch.device('cpu')

    # get data
    print("Preparing Data\n")
    data = pickle.load(open("puzzle8_data.pkl", "rb"))

    states_nnet, outputs = sample_training_data(data['states_nnet'], data['outputs'], int(1e6))

    # train with supervised learning
    print("Training DNN\n")
    nnet.train()
    train_nnet(nnet, states_nnet, outputs)

    # get performance
    print("Evaluating DNN\n")
    nnet.eval()
    out_nnet = nnet(torch.tensor(data["states_nnet"].astype(np.float32), device=device).float()).cpu().data.numpy()
    mse = float(np.mean((out_nnet - data["outputs"]) ** 2))
    for cost_to_go in np.unique(data["outputs"]):
        idxs_targ: np.array = np.where(data["outputs"] == cost_to_go)[0]
        states_targ_nnet: np.ndarray = data["states_nnet"][idxs_targ]

        out_nnet = nnet(torch.tensor(states_targ_nnet, device=device).float()).cpu().data.numpy()

        mse = float(np.mean((out_nnet - cost_to_go) ** 2))
        print("Cost-To-Go: %i, Ave DNN Output: %f, MSE: %f" % (cost_to_go, float(np.mean(out_nnet)), mse))

    print("Total MSE: %f" % mse)


if __name__ == "__main__":
    main()
