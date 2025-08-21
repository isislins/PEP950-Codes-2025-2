from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer
from qiskit import transpile

# Create a new circuit with three qubits and three classical bits
qc = QuantumCircuit(2, 2)

# --- State Preparation ---
qc.h(0)
qc.h(1)
# com ambos os qubits em superposição, temos |00> + |01> + |10> + |11>
# --- Oracle (|10> marked em bits isso é 01 pois os qubits são invertidos na medição) ---
qc.x([1]) # Troco o 0 por 1

## porta CZ
qc.h(1) # coloco em superposição o último qubit
qc.cx(0, 1) # Inverte o sinal do estado |11> (A ideia é que o cx entre duas portas H é um CZ, e o CZ inverte o sinal do estado |111...1>)
qc.h(1)
## fim da porta CZ
qc.x([1]) # volto o último qubit para |0>
# inversão de fase para o estado marcado

# --- Grover Operator ---
qc.h([0, 1])
qc.x([0, 1])
qc.h(1)
qc.cx(0, 1)
qc.h(1)
qc.x([0,1])
qc.h([0,1])


# Visualizar o circuito
print(qc.draw('text'))


# --- Measurement ---
qc.measure([0, 1], [0, 1])

# Para obter o statevector:
state_simulator = Aer.get_backend('statevector_simulator')
qc_t = transpile(qc, state_simulator)
state_result = state_simulator.run(qc_t).result()
statevector = state_result.get_statevector(qc_t)
print("Statevector:", statevector)

# Para obter as contagens (medidas):
simulator = Aer.get_backend('qasm_simulator')
qc_t2 = transpile(qc, simulator)
result = simulator.run(qc_t2, shots=1024).result()
counts = result.get_counts(qc_t2)
print("Resultados das medições:", counts)
