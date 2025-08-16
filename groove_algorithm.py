from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer
from qiskit import transpile

# Create a new circuit with three qubits and three classical bits
qc = QuantumCircuit(3, 3)

# --- State Preparation ---
qc.x(0)  # Porta X no q0
qc.h(0)
qc.h(1)
qc.h(2)

# --- Toffoli Gate (CCX decomposta em portas elementares) ---
qc.h(0)
qc.cx(1, 0)
qc.tdg(0)
qc.cx(2, 0)
qc.t(0)
qc.cx(1, 0)
qc.tdg(0)
qc.cx(2, 0)
qc.t(0)
qc.tdg(1)
qc.h(0)
qc.cx(2, 1)
qc.tdg(1)
qc.cx(2, 1)
qc.s(1)
qc.t(2)

# --- Grover Operator ---
qc.h([1, 2])
qc.x([1, 2])
qc.h(1)
qc.cx(2, 1)
qc.h(1)
qc.x(2)
qc.h(2)
qc.x(1)
qc.h(1)


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
