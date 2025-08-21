from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Criamos um circuito de 2 qubits
qc = QuantumCircuit(2)

# Preparamos o estado |+> no primeiro qubit (controle)
qc.h(0)

# O segundo qubit já está em |0> por padrão

# Aplicamos o CNOT (controle = qubit 0, alvo = qubit 1)
qc.cx(0, 1)

# Mostramos o circuito
print(qc.draw())

# Obtemos o vetor de estado resultante
state = Statevector.from_instruction(qc)
print("Statevector final:", state)

# Mostramos os amplitudes de cada estado computacional
state_dict = state.to_dict()
print("\nAmplitudes por base computacional:")
for k, v in state_dict.items():
    print(f"{k}: {v}")