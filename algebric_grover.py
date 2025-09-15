## Gustavo Camargo - 2025
## PEP950 - Quantum Computing
# Algebric implementation of Grover's algorithm for n qubits
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
steps = 0
nQubits = 4
oracle = [identity,identity,identity,identity] # oracle to mark |000...0> state


def getCZ(n):
    CZ = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(CZ.shape[0]):
        CZ[i][i] = 1
    CZ[CZ.shape[0]-1][CZ.shape[0]-1] = -1
    return CZ
    
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

# Create a function that apply a specific gate to specific qubits in a n-qubit system
def getSpecificTransformation(gates):
    result = gates[0]
    for gate in gates[1:]:
        result = tensor_product(result, gate)
    return result

# Create a getTransformation function that apply a specific gate to all qubits
def getTransformation(gate, n_qubits):  
    result = gate
    for _ in range(n_qubits - 1):
        result = tensor_product(result, gate)
    return result


def grover(statevector, steps):
    #for i in np.pi/4 * np.sqrt(2**nQubits):
    steps += 1
    bestProb = 0
    ## Apply grover

    ## Apply X to all |0> qubits of desired state to prepare for oracle
    statevector = getSpecificTransformation(oracle) @ statevector
    ## Apply CZ gate as oracle to mark state |111...1>
    statevector = getCZ(nQubits) @ statevector
    ## Apply X to all |0> qubits of desired state
    statevector = getSpecificTransformation(oracle) @ statevector
    ## Apply Diffusion operator
    statevector = D @ statevector
    print(f"Statevector after Diffusion {steps} time:\n {statevector}")
    for i in range(statevector.shape[0]):
        p = np.abs(statevector[i][0])**2
        if p >= 0.95:
            bestProb = p
            print(f"Best state found: |{format(i, f'0{nQubits}b')}> with probability {p} after {steps} steps")
            return
    if bestProb < 0.95:
        grover(statevector, steps)



# grover for n qubits
statevector = np.zeros((2**nQubits,1), dtype=complex )  # initial state |00>
statevector[0][0] = 1  # set the initial state to |00>
D, W, R = diffusion_operator(nQubits)
print("Diffusion Operator D:\n", D)
#print("W Matrix:\n", W)
#print("R Matrix:\n", R)
#print("Initial Statevector:\n", statevector)
# Apply Hadamard to all qubits to create superposition
H_all = getTransformation(gateH, nQubits)
#print(H_all)
statevector = H_all @ statevector
print("Statevector after Hadamard:\n", statevector)

statevector = getCZ(nQubits) @ statevector
print("Statevector after Oracle (CZ):\n", statevector)

statevector = getSpecificTransformation(oracle) @ statevector
print("Statevector after Oracle (X-CZ-X):\n", statevector)

statevector = D @ statevector
print("Statevector after Diffusion:\n", statevector)






#grover(statevector,steps) #pi/4 * sqrt(N) iterations should be enough [8 = 12; 12 = 50]


        





  


