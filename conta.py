#make a tensor product of two matrices
import numpy as np  

# Function to create a tensor product of two matrices
def tensor_product(mat1, mat2):
    return np.kron(mat1, mat2)  
# Example usage:
A = np.array([[1, 2], [3, 4]])  
B = np.array([[0, 5], [6, 7]])
result = tensor_product(A, B)
print("Tensor Product of A and B:\n", result)


