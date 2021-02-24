import torch


def print_tensor(x, name=None):
    if name is not None:
        print(name)

    print("Type          :", x.type())
    print("Shape/size    :", x.shape)
    print("Requires grad :", x.requires_grad)
    print("Values        :\n", x, "\n")


# How to create a tensor
def tensor_creation():
    print("Tensor creation section")
    # Create empty tensor 3x4
    A = torch.empty(3, 4)

    # Creation a 3x4 tensor with random values from a random uniform distribution
    # random values from range 0 <= x < 1
    B1 = torch.rand(3, 4)

    # Creation a 3x4 tensor with random values from a random normal distribution
    B2 = torch.randn(3, 4)

    # Create tensor 3x4 with random values and specific data type
    B3 = torch.rand(3, 4, dtype=torch.double)

    # Create tensor 3x4 with zeros
    C = torch.zeros(3, 4)

    # Create tensor 3x4 with ones
    D = torch.ones(3, 4)

    # Create tensor from values
    E1 = torch.tensor(3)
    E2 = torch.tensor([5, 7, 8.9])
    E3 = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Create identity matrix
    F = torch.zeros(5, 5)
    for i in range(5):
        F[i, i] = 1

    print_tensor(A, name="Tensor A (empty)")
    print_tensor(B1, name="Tensor B1 (with random values from a random uniform distribution)")
    print_tensor(B2, name="Tensor B2 (with random values from a random normal distribution)")
    print_tensor(B3, name="Tensor B3 (with random values and double data type)")
    print_tensor(C, name="Tensor C (with zeros)")
    print_tensor(D, name="Tensor E (with ones)")
    print("\nTensors created from values")
    print_tensor(E1, name="Tensor E1")
    print_tensor(E2, name="Tensor E2")
    print_tensor(E3, name="Tensor E3")
    print_tensor(F, name="Tensor F")


# How to modify a tensor
def modify_tensor():
    print("Tensor modify section")
    # How to reshape tensor
    x_original = torch.ones(3, 4)
    x_reshaped = x_original.view(2, 6)

    # How to create numpy array from tensor
    x_numpy = x_original.numpy()

    print_tensor(x_original, name="X original")
    print_tensor(x_reshaped, name="X reshaped")
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

    print_tensor(x1, name="Tensor X1")
    print_tensor(y1, name="Tensor Y1")
    print_tensor(z1, name="Tensor Z1(sum of tensors X1 and Y1)")
    print_tensor(x2, name="Tensor X2")


if __name__ == "__main__":
    print("Application started")
    print("Torch version : ", torch.__version__)
    tensor_creation()
    #modify_tensor()
    #tensor_operations()

    print("Success")