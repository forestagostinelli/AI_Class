from typing import List
import numpy as np

import torch
from torch import nn

from environments.n_puzzle import NPuzzle, NPuzzleState

import pickle

from proj_code.proj_grad import get_nnet_model, train_nnet, value_iteration


def evaluate_dnn(nnet, device, data):
    # get performance
    print("Evaluating DNN\n")
    nnet.eval()
    out_nnet_neg = -nnet(torch.tensor(data["states_nnet"].astype(np.float32), device=device).float()).cpu().data.numpy()

    mse = float(np.mean((out_nnet_neg - data["outputs"]) ** 2))
    for cost_to_go in np.unique(data["outputs"]):
        idxs_targ: np.array = np.where(data["outputs"] == cost_to_go)[0]
        states_targ_nnet: np.ndarray = data["states_nnet"][idxs_targ]

        out_nnet_neg = -nnet(torch.tensor(states_targ_nnet, device=device).float()).cpu().data.numpy()

        mse = float(np.mean((out_nnet_neg - cost_to_go) ** 2))
        print("Cost-To-Go: %i, Ave Negative DNN Output: %f, MSE: %f" % (cost_to_go, float(np.mean(out_nnet_neg)), mse))

    print("Total MSE: %f" % mse)


def main():
    torch.set_num_threads(1)

    # get environment
    env: NPuzzle = NPuzzle(3)

    # get nnet model
    nnet: nn.Module = get_nnet_model()
    device = torch.device('cpu')
    num_generate: int = 20000
    num_vi_updates: int = 50

    # get data
    data = pickle.load(open("puzzle8_data.pkl", "rb"))

    # train with supervised learning
    print("Training DNN\n")
    for vi_update in range(num_vi_updates):
        print("--- Value Iteration Update: %i ---" % vi_update)
        states: List[NPuzzleState] = env.generate_states(num_generate, (0, 500))

        outputs_np = value_iteration(nnet, env, states)
        outputs = np.expand_dims(np.array(outputs_np), 1).astype(np.float32)

        nnet.train()
        states_nnet: np.ndarray = env.states_to_nnet_input(states)
        train_nnet(nnet, states_nnet, outputs)

        nnet.eval()
        evaluate_dnn(nnet, device, data)


if __name__ == "__main__":
    main()
