from proj_code.proj3 import relu_forward, relu_backward, linear_forward, linear_backward
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--layer', type=str, required=True, help="linear,relu")
    parser.add_argument('--num_inputs', type=int, default=5, help="")
    parser.add_argument('--input_dim', type=int, default=10, help="")
    parser.add_argument('--hidden_dim', type=int, default=100, help="")
    args = parser.parse_args()

    device: torch.device = torch.device("cpu")

    inputs = np.random.normal(size=(args.num_inputs, args.input_dim)).astype(np.float32)
    inputs_torch = Variable(torch.tensor(inputs), requires_grad=True)
    if args.layer.lower() == "linear":
        # initialize
        weights = np.random.normal(size=(args.hidden_dim, args.input_dim)).astype(np.float32)
        biases = np.zeros(args.hidden_dim).astype(np.float32)

        # pytorch forward
        linear_torch = nn.Linear(args.input_dim, args.hidden_dim)
        linear_torch.weight.data = torch.tensor(weights, device=device)
        linear_torch.bias.data = torch.tensor(biases, device=device)

        out_torch = linear_torch(inputs_torch)

        # numpy forward
        out_np = linear_forward(inputs, weights, biases)
        shape = out_np.shape
        target_shape = (args.num_inputs, args.hidden_dim)
        assert shape == target_shape, "Output shape is %s instead of %s" % (str(shape), str(target_shape))

        # error forward
        sq_err = np.mean((out_torch.cpu().data.numpy() - out_np) ** 2)
        print("Squared error for output: %f" % sq_err)

        # pytorch backward
        loss_torch = torch.sum(out_torch ** 2)
        loss_torch.backward()

        inputs_grad_torch = inputs_torch.grad
        weights_grad_torch = linear_torch.weight.grad
        biases_grad_torch = linear_torch.bias.grad

        # numpy backward
        grad = 2 * out_np
        weights_grad_np, biases_grad_np, inputs_grad_np = linear_backward(grad, inputs, weights)

        # error backward
        shape = weights_grad_np.shape
        target_shape = weights.shape
        assert shape == target_shape, "Weights grad shape is %s instead of %s" % (str(shape), str(target_shape))

        sq_err = np.mean((weights_grad_torch.cpu().data.numpy() - weights_grad_np) ** 2)
        print("Squared error for gradient w.r.t. weights: %f" % sq_err)
        if sq_err > 1e-10:
            print("ERROR")

        shape = biases_grad_np.shape
        target_shape = biases.shape
        assert shape == target_shape, "Biases grad shape is %s instead of %s" % (str(shape), str(target_shape))

        sq_err = np.mean((biases_grad_torch.cpu().data.numpy() - biases_grad_np) ** 2)
        print("Squared error for gradient w.r.t. biases: %f" % sq_err)
        if sq_err > 1e-10:
            print("ERROR")

        shape = inputs_grad_np.shape
        target_shape = inputs.shape
        assert shape == target_shape, "Inputs grad shape is %s instead of %s" % (str(shape), str(target_shape))

        sq_err = np.mean((inputs_grad_torch.cpu().data.numpy() - inputs_grad_np) ** 2)
        print("Squared error for gradient w.r.t. inputs: %f" % sq_err)
        if sq_err > 1e-10:
            print("ERROR")

    elif args.layer.lower() == "relu":
        # pytorch forward
        out_torch = torch.relu(inputs_torch)

        # numpy forward
        out_np = relu_forward(inputs)
        shape = out_np.shape
        target_shape = (args.num_inputs, args.input_dim)
        assert shape == target_shape, "Output shape is %s instead of %s" % (str(shape), str(target_shape))

        # error forward
        sq_err = np.mean((out_torch.cpu().data.numpy() - out_np) ** 2)
        print("Squared error for output: %f" % sq_err)
        if sq_err > 1e-10:
            print("ERROR")

        # pytorch backward
        loss_torch = torch.sum((out_torch + 2) ** 2)
        loss_torch.backward()

        inputs_grad_torch = inputs_torch.grad

        # numpy backward
        grad = 2 * (out_np + 2)
        inputs_grad_np = relu_backward(grad, inputs)

        # pytorch backward

        # error backward
        shape = inputs_grad_np.shape
        target_shape = inputs.shape
        assert shape == target_shape, "Inputs grad shape is %s instead of %s" % (str(shape), str(target_shape))

        sq_err = np.mean((inputs_grad_torch.cpu().data.numpy() - inputs_grad_np) ** 2)
        print("Squared error for gradient w.r.t. inputs: %f" % sq_err)
        if sq_err > 1e-10:
            print("ERROR")

    else:
        raise ValueError("Unknown layer type %s" % args.layer)


if __name__ == "__main__":
    main()
