# qaoa_min_example.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Diagonal
from qiskit_algorithms.optimizers import COBYLA
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_algorithms import QAOA

# --- 1) Definir rotas e custos ---
# Mapeamento de base (2 qubits): '00','01','10','11'
route_costs = {
    "00": 10.0,   # rota A
    "01": 7.0,    # rota B (ótima)
    "10": 15.0,   # rota C
    "11": 100.0   # estado inválido/penalizado
}


# --- 2) Construir circuito QAOA (p = 1) parametrizado por (gamma, beta) ---
def qaoa_circuit(gamma, beta):
    qr = QuantumRegister(2, 'q')
    qc = QuantumCircuit(qr)

    # inicialização em superposição uniforme
    qc.h(qr)

    # Phase-separator: aplicar rotações condicionais que representam exp(-i*gamma*cost)
    # Aqui custo é só um deslocamento de fase por estado |00>,|01>,|10>,|11|
    # Implementamos com uma sequência de CZ+RZ simples
    qc.rz(gamma * route_costs["00"], [qr[0]])  # rota "00" -> fase em q0 quando ambos=0
    qc.cz(qr[0], qr[1])
    qc.rz(gamma * (route_costs["11"] - route_costs["00"]), qr[1])
    qc.cz(qr[0], qr[1])

    # (para um exemplo didático, esse é um atalho: para problemas reais
    #  usamos opflow/PauliSumOp para gerar os exponentials do Hamiltoniano)

    # Mixer: RX(2*beta) em cada qubit
    qc.rx(2*beta, qr[0])
    qc.rx(2*beta, qr[1])

    return qc

# --- 3) Função objetivo: expectativa do custo dado (gamma,beta) ---
def expected_cost(gamma_beta):
    gamma, beta = gamma_beta
    qc = qaoa_circuit(gamma, beta)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()  # dict: {'00':p, '01':p, ...}
    exp = 0.0
    for basis, p in probs.items():
        exp += p * route_costs[basis]
    return exp

# --- 4) Otimização ---
opt = COBYLA(maxiter=200)
# inicial guess (gamma, beta)
x0 = np.array([0.8, 0.8])  

res = opt.minimize(fun=expected_cost, x0=x0)
gamma_opt, beta_opt = res.x
print("Resultado otimização:", res)
print(f"gamma_opt = {gamma_opt:.4f}, beta_opt = {beta_opt:.4f}")
print("Expectativa de custo otimizada:", expected_cost([gamma_opt, beta_opt]))

# --- 5) Gerar circuito final e amostrar ---
qc_final = qaoa_circuit(gamma_opt, beta_opt)
cr = ClassicalRegister(2, 'c')
qc_final.add_register(cr)
qc_final.measure(qc_final.qregs[0], cr)


backend = Aer.get_backend('qasm_simulator')
shots = 2000
job = backend.run(qc_final, shots=shots)
counts = job.result().get_counts()
print("Contagens de medição (QAOA p=1):", counts)

# plot histogram
plt.figure(figsize=(6,4))
plot_histogram(counts)
plt.title("QAOA (p=1) - distribuição final")
plt.show()
