# grover_simple.py
# Versão (A): Oracle clássico via máscara de rotas "good"
# Execute com: python grover_simple.py

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# Dados toy: 8 rotas, cada uma com 3 atributos binários
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

def grover_with_classical_mask(good_mask, shots=2000):
    n = int(np.log2(len(good_mask)))
    idx = QuantumRegister(n, 'idx')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(idx, cr)

    # Superposição inicial
    qc.h(idx)

    # Oracle
    for i, is_good in enumerate(good_mask):
        if is_good:
            bits = [(i >> j) & 1 for j in range(n)]
            for qpos, b in enumerate(bits):
                if b == 0:
                    qc.x(idx[qpos])
            qc.h(idx[n-1])
            if n-1 == 0:
                qc.z(idx[0])
            else:
                qc.mcx(list(range(n-1)), idx[n-1])
            qc.h(idx[n-1])
            for qpos, b in enumerate(bits):
                if b == 0:
                    qc.x(idx[qpos])

    # Difusão
    qc.h(idx)
    qc.x(idx)
    qc.h(idx[n-1])
    if n-1 == 0:
        qc.z(idx[0])
    else:
        qc.mcx(list(range(n-1)), idx[n-1])
    qc.h(idx[n-1])
    qc.x(idx)
    qc.h(idx)

    qc.measure(idx, cr)
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=2000)
    result = job.result()
    return result.get_counts()

if __name__ == "__main__":
    counts = grover_with_classical_mask(good_mask)
    print("Resultados:", counts)
    plt.bar(counts.keys(), counts.values())
    plt.title("Grover - Oracle clássico (máscara)")
    plt.show()
