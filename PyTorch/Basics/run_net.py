import torch
import torch.nn as nn


class ComputeSum(nn.Module):
    def __init__(self):
        super(ComputeSum, self).__init__()

    def forward(self, x, y):
        return x.add(y)


def main():
    print("Application started")
    print("Torch version : ", torch.__version__)

    # Load model weights
    #model_filename = "compute_sum.pt"
    #model = ComputeSum()
    #model.load_state_dict(torch.load(model_filename))

    # Load entire model
    # Model class must be defined somewhere
    #model_filename = "full_compute_sum.pt"
    #model = torch.load(model_filename)

    # Load serialized model
    model = torch.jit.load("traced_compute_sum.pt")

    x1 = torch.tensor([[1, 1, 1], [0, 0, 0]])
    y1 = torch.tensor([[0, 0, 0], [2, 2, 2]])
    result = model(x1, y1)
    print("Result :\n", result)
    print("shape : ", result.shape)

    print("Success")


if __name__ == "__main__":
    main()
