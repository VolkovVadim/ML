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

    x1 = torch.tensor([[1, 1, 1], [0, 0, 0]])
    y1 = torch.tensor([[0, 0, 0], [1, 1, 1]])
    model = ComputeSum()
    result = model(x1, y1)
    print("Result :\n", result)

    print("Saving model to file...")
    # Save model weights
    #torch.save(model.state_dict(), "compute_sum.pt")

    # Save entire model
    torch.save(model, "full_compute_sum.pt")

    # Serialize model
    #model.eval()
    #traced_script_module = torch.jit.trace(model, [x1, y1])
    #traced_script_module.save("traced_compute_sum.pt")

    print("Success")


if __name__ == "__main__":
    main()
