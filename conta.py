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
    #if np.iscomplex(gate).any() == True:
    mat = np.zeros((2*n_qubits, 2*n_qubits), dtype=complex)
    #else:
    #    mat = np.zeros((2*n_qubits, 2*n_qubits))
    if qubit_index >= n_qubits:
        raise ValueError("Qubit index out of range")
    
    for i in range(0, mat.shape[0], 2):
            if i == qubit_index*2:
                mat[i][i] = gate[0][0]
                mat[i][i+1] = gate[0][1]
                mat[i+1][i] = gate[1][0]
                mat[i+1][i+1] = gate[1][1]
            else:
                mat[i][i] = 1
                mat[i+1][i+1] = 1
    return mat

# Create a getTransformation function that apply a specific gate to all qubits
def getTransformation(gate, n_qubits):  
    result = gate
    for _ in range(n_qubits - 1):
        result = tensor_product(result, gate)
    return result

# grover for 2 qubits
statevector = np.array([[1], [0], [0], [0]], dtype=complex )  # initial state |00>
D, W, R = diffusion_operator(np.sqrt(len(statevector)).astype(int))
print("Diffusion Operator D:\n", D)
print("W Matrix:\n", W)
print("R Matrix:\n", R)
print("Initial Statevector:\n", statevector)
# Apply Hadamard to all qubits to create superposition
H_all = getTransformation(gateH, 2)
statevector = H_all @ statevector
print("Statevector after Hadamard:\n", statevector)
# Apply Oracle (for example, marking state |10>, which is index 2)
oracle = getSpecificTransformation(gateX, 0, 2)  # X gate on qubit 0 to flip |00> to |10>
print("oracleX:\n", oracle)
statevector = oracle @ statevector
print("Statevector after Oracle:\n", statevector)

#print(getSpecificTransformation(gateS,1,2))

#result = tensor_product(A, B)
#print("Tensor Product of A and B:\n", result)
