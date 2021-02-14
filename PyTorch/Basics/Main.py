import torch


if __name__ == "__main__":
    print("Application started")
    print("Torch version : ", torch.__version__)

    # Create empty tensor 3x4
    A = torch.empty(3, 4)

    # Create tensor 3x4 with random values
    B = torch.rand(3, 4)

    print("Tensor A (empty):\n", A)
    print("Tensor B (with random values):\n", B)

    print("Success")