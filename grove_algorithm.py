from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer
from qiskit import transpile

# Create a new circuit with three qubits and three classical bits
qc = QuantumCircuit(3, 3)

# --- State Preparation ---
qc.h([0,1,2])
# com ambos os qubits em superposição, temos |00> + |01> + |10> + |11>
# --- Oracle (|010> marked em bits isso é 101 pois os qubits são invertidos na medição) ---
qc.x([0,1]) # Troco o 0 por 1

## porta CZ para inverter a fase do estado marcado
qc.h(2) # coloco em superposição o último qubit
qc.mcx([0, 1],2) # Inverte o sinal do estado |111> (A ideia é que o cx entre duas portas H é um CZ, e o CZ inverte o sinal do estado |111...1>)
qc.h(2)
## fim da porta CZ
qc.x([0,1]) # retorno os qubits trocados na entrada do oráculo para o estad


# --- Grover Operator ---
#qc.h([0, 1])
#qc.x([0, 1])
#qc.h(1)
#qc.mcx([0], 1)
#qc.h(1)
#qc.x([0,1])
#qc.h([0,1])

qc.h([0, 1])
qc.z([0, 1])
qc.cz(0,1)
qc.h([0,1])

# Visualizar o circuito
print(qc.draw('text'))


# --- Measurement ---
qc.measure([0, 1, 2], [0, 1, 2])

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
