from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer
from qiskit import transpile

nQubits = 4
# Create a new circuit with three qubits and three classical bits
qc = QuantumCircuit(nQubits, nQubits)

# --- State Preparation ---
qc.h(list(range(nQubits)))
print(list(range(nQubits)))
# com ambos os qubits em superposição, temos |00> + |01> + |10> + |11>
# --- Oracle (|010> marked em bits isso é 101 pois os qubits são invertidos na medição) ---
#qc.x([0,1]) # Troco o 0 por 1

## porta CZ para inverter a fase do estado marcado
qc.h(nQubits-1) # coloco em superposição o último qubit
qc.mcx(list(range(nQubits-1)),nQubits-1) # Inverte o sinal do estado |111> (A ideia é que o cx entre duas portas H é um CZ, e o CZ inverte o sinal do estado |111...1>)
qc.h(nQubits-1)
## fim da porta CZ
qc.x([0,1,2,3]) # retorno os qubits trocados na entrada do oráculo para o estad


# --- Grover Operator ---
#qc.h([0, 1])
#qc.x([0, 1, 2])
#qc.h(2)
#qc.mcx([0,1], 2)
#qc.h(2)
#qc.x([0,1, 2])
#qc.h([0,1,2])

qc.h(list(range(nQubits)))
qc.z(list(range(nQubits)))
qc.cz(list(range(nQubits-1)),nQubits-1)
qc.h(list(range(nQubits)))

# Visualizar o circuito
print(qc.draw('text'))


# --- Measurement ---
qc.measure(list(range(nQubits)), list(range(nQubits)))

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
