import torch


# How to create a tensor
def tensor_creation():
    print("Tensor creation section")
    # Create empty tensor 3x4
    A = torch.empty(3, 4)

    # Create tensor 3x4 with random values
    B = torch.rand(3, 4)

    # Create tensor 3x4 with random values and specific data type
    C = torch.rand(3, 4, dtype=torch.double)

    # Create tensor 3x4 with zeros
    D = torch.zeros(3, 4)

    # Create tensor 3x4 with ones
    E = torch.ones(3, 4)

    # Create tensor from values
    F1 = torch.tensor(3)
    F2 = torch.tensor([5, 7, 8.9])
    F3 = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Create identity matrix
    G = torch.zeros(5, 5)
    for i in range(5):
        G[i, i] = 1

    print("Tensor A (empty):\n", A)
    print("Tensor B (with random values):\n", B)
    print("Tensor C (with random values and double data type):\n", C)
    print("Tensor D (with zeros):\n", D)
    print("Tensor E (with ones):\n", E)
    print("\nTensors created from values")
    print("Tensor F1:\n", F1)
    print("Tensor F2:\n", F2)
    print("Tensor F3:\n", F3, "\n")
    print("Tensor G:\n", G)


# How to modify a tensor
def modify_tensor():
    print("Tensor modify section")
    # How to reshape tensor
    x_original = torch.ones(3, 4)
    x_reshaped = x_original.view(2, 6)

    # How to create numpy array from tensor
    x_numpy = x_original.numpy()

    print("X original:\n", x_original)
    print("shape : ", x_original.size())
    print("X reshaped:\n", x_reshaped)
    print("shape :", x_reshaped.size())
    print("X original as numpy array:\n", x_numpy)


# Operations with tensor
def tensor_operations():
    print("Tensor operations section")
    # Addition of tensors
    x1 = torch.tensor([[1, 1, 1], [0, 0, 0]])
    y1 = torch.tensor([[0, 0, 0], [1, 1, 1]])
    z1 = x1 + y1
    # another way
    # z1 = torch.add(x1, y1)
    # or
    # z1 = x1.add(y1)

    x2 = torch.tensor([[1, 1, 1], [0, 0, 0]])
    y2 = torch.tensor([[1, 1, 1], [1, 1, 1]])
    x2.add_(y2)  # not to be confused with the non-underscore method

    print("Tensor X1:\n", x1)
    print("Tensor Y1:\n", y1)
    print("Tensor Z1(sum of tensors X1 and Y1):\n", z1)
    print("Tensor X2:\n", x2)


if __name__ == "__main__":
    print("Application started")
    print("Torch version : ", torch.__version__)
    #tensor_creation()
    #modify_tensor()
    tensor_operations()

    print("Success")