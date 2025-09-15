# grover_qrac_vqc.py
# Versão (B): Oracle via encoding QRAC-like + VQC treinado
# Execute com: python grover_qrac_vqc.py

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
import matplotlib.pyplot as plt

# Dados toy: 8 rotas com 3 atributos binários
route_attributes = {
    0: [0,0,1],
    1: [0,1,1],
    2: [1,0,1],
    3: [0,0,0],
    4: [1,1,1],
    5: [0,1,0],
    6: [1,0,0],
    7: [1,1,0]
}

def classical_good(attr):
    return (attr[0] == 0) and (attr[1] == 0) and (attr[2] == 1)

good_mask = [int(classical_good(route_attributes[i])) for i in range(8)]
print("Good mask:", good_mask)

# Função para preparar feature qubit (QRAC-like encoding)
def qrac_3_to_1_angles(bits):
    b0,b1,b2 = bits
    theta = (b0*1.57 + b1*0.78 + b2*0.39)
    phi   = (b0*0.0 + b1*1.2 + b2*2.4)
    return theta, phi

# Probabilidade ancilla=1 dado índice e params
def ancilla_prob_for_index(idx_value, params):
    qc = QuantumCircuit(2)  # [feature, ancilla]
    bits = route_attributes[idx_value]
    theta, phi = qrac_3_to_1_angles(bits)
    qc.rz(phi, 0)
    qc.ry(theta, 0)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    qc.crx(params[2], 0, 1)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    prob_anc1 = sum(p for basis, p in probs.items() if basis[0] == '1')
    return prob_anc1

def objective(params):
    eps = 1e-10
    loss = 0.0
    for i in range(8):
        p = ancilla_prob_for_index(i, params)
        y = good_mask[i]
        loss += - (y * np.log(p+eps) + (1-y)*np.log(1-p+eps))
    return loss / 8.0

# Treino VQC
opt = COBYLA(maxiter=1000, tol=1e-4)
initial = np.random.randn(3) * 0.1
res = opt.minimize(fun=objective, x0=initial)
trained_params = res.x
print("Treinamento concluído. Params:", trained_params, "Loss:", res.fun)

# Construir oracle (simplificado, não totalmente reversível)
def build_oracle(trained_params):
    idx = QuantumRegister(3, 'idx')
    feat = QuantumRegister(1, 'feat')
    mark = QuantumRegister(1, 'mark')
    qc = QuantumCircuit(idx, feat, mark, name='oracle')

    qc.h(mark)
    qc.z(mark)

    # preparar feature approx (controlado simplificado)
    for i in range(3):
        qc.crz(0.5, idx[i], feat[0])
        qc.cry(0.5, idx[i], feat[0])

    qc.ry(trained_params[0], feat[0])
    qc.rz(trained_params[1], feat[0])
    qc.crx(trained_params[2], feat[0], mark[0])

    qc.z(mark)
    qc.crx(-trained_params[2], feat[0], mark[0])
    qc.rz(-trained_params[1], feat[0])
    qc.ry(-trained_params[0], feat[0])

    qc.z(mark)
    qc.h(mark)
    return qc

oracle = build_oracle(trained_params)

def grover_with_oracle(oracle_circ, shots=2000):
    idx = QuantumRegister(3, 'idx')
    feat = QuantumRegister(1, 'feat')
    mark = QuantumRegister(1, 'mark')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(idx, feat, mark, cr)

    qc.h(idx)
    qc.compose(oracle_circ, inplace=True)

    qc.h(idx)
    qc.x(idx)
    qc.h(idx[2])
    qc.mcx([idx[0], idx[1]], idx[2])
    qc.h(idx[2])
    qc.x(idx)
    qc.h(idx)

    qc.measure(idx, cr)
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=2000)
    result = job.result()
    return result.get_counts()

if __name__ == "__main__":
    counts = grover_with_oracle(oracle)
    print("Resultados (oracle VQC):", counts)
    plt.bar(counts.keys(), counts.values())
    plt.title("Grover - Oracle via QRAC-like + VQC")
    plt.show()
