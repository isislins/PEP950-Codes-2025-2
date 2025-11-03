from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer import Aer

# --- 1. Definir rotas e custos ---
route_costs = {
    "00": 10,
    "01": 7,
    "10": 15
}
# rota ótima = "01" (custo 7)

# --- 2. Construir oracle baseado em threshold ---
def build_threshold_oracle(L):
    idx = QuantumRegister(2, 'idx')
    mark = QuantumRegister(1, 'mark')
    qc = QuantumCircuit(idx, mark, name='oracle')

    # Inicializar mark qubit em estado |->
    qc.h(mark)
    qc.z(mark)

    # Marcar estados cujo custo <= L
    for route, cost in route_costs.items():
        if cost <= L:
            controls = [idx[i] for i, bit in enumerate(route[::-1]) if bit == '1']
            inversions = [i for i, bit in enumerate(route[::-1]) if bit == '0']

            # inverter qubits que deveriam ser 0
            for i in inversions:
                qc.x(idx[i])

            # aplicar multi-controlled Z
            if len(controls) == 0:
                qc.z(mark)
            elif len(controls) == 1:
                qc.cz(controls[0], mark)
            else:
                qc.mcx(controls, mark)

            # desfazer inversões
            for i in inversions:
                qc.x(idx[i])

    qc.h(mark)
    return qc

# --- 3. Grover iteration ---
def grover_iteration(oracle):
    idx = QuantumRegister(2, 'idx')
    mark = QuantumRegister(1, 'mark')
    qc = QuantumCircuit(idx, mark)

    # inicializar idx em superposição
    qc.h(idx)

    # aplicar oracle
    qc.append(oracle, qc.qubits)

    # difusor
    qc.h(idx)
    qc.x(idx)
    qc.h(idx[1])
    qc.cx(idx[0], idx[1])
    qc.h(idx[1])
    qc.x(idx)
    qc.h(idx)

    return qc

# --- 4. Rodar Grover com diferentes thresholds ---
def run_grover(L, shots=1000):
    oracle = build_threshold_oracle(L)
    qc = grover_iteration(oracle)
    cr = ClassicalRegister(2, 'c')
    qc.add_register(cr)
    qc.measure(qc.qregs[0], cr)

    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc, shots=2000)
    result = job.result()
    return  result.get_counts()


# --- 5. Testar com limites ---
results_12 = run_grover(L=12)   # deve marcar "00" e "01"
results_8  = run_grover(L=7)    # deve marcar apenas "01"

print("Threshold L=12:", results_12)
print("Threshold L=8:", results_8)

# plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plot_histogram(results_12, ax=plt.gca(), title="Grover com L=12")
plt.subplot(1,2,2)
plot_histogram(results_8, ax=plt.gca(), title="Grover com L=8")
plt.show()
