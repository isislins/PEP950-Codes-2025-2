#make a tensor product of two matrices
import numpy as np  

# Function to create a tensor product of two matrices
def tensor_product(mat1, mat2):
    return np.kron(mat1, mat2)  
# quantum gates:
identity = np.array([[1, 0], [0, 1]])
gateX = np.array([[0, 1], [1, 0]])  
gateY = np.array([[0, -1j], [1j, 0]])
gateZ = np.array([[1, 0], [0, -1]])
gateH = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
gateS = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]])
gateT = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
nQubits = 2

# function to create a diffusion operator for Grover's algorithm where D = W*R*W where Wij = 2^(-n/2) * (-1)^(i.j)
# and (i,j) are the binary representations of the row and column indices. And R is a diagonal matrix with R[0,0] = 1 and R[i,i] = -1 for i > 0.
def diffusion_operator(n):
    size = 2 ** n
    W = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            W[i, j] = (1 / np.sqrt(size)) * (-1) ** (bin(i & j).count('1'))
    R = np.diag([1] + [-1] * (size - 1))
    D = W @ R @ W
    return D, W, R

# given a number of qubits, if i want to apply a gate to a specific qubit, I need to create a matrix that allows me to take an array of qubits and apply the gate to the specific qubit while leaving the others unchanged.
# Example: if we have 2 qubits and we want to apply gateX to the first qubit (index 0), then the resulting matrix will: [[0, 1, 0, 0], [1, 0, 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]]
# If we want to apply gateX to the second qubit (index 1), then the resulting matrix will: [[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0]]
def getSpecificTransformation(gate, qubit_index, n_qubits):
    mat = np.zeros((2**n_qubits, 2**n_qubits))
    if qubit_index >= n_qubits:
        raise ValueError("Qubit index out of range")
    for i in range(0, mat.shape[0], 2):
        if i == 0:
            result = gate if i == qubit_index else identity
        else:
            result = tensor_product(result, gate if i == qubit_index else identity)



#result = tensor_product(A, B)
#print("Tensor Product of A and B:\n", result)
