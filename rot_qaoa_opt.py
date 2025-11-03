# rot_qaoa_p.py
import numpy as np
import matplotlib.pyplot as plt
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit.visualization import plot_histogram

# --- 1) Definir problema de roteirização simplificado ---
# 3 rotas possíveis: A (10), B (7), C (15). "B" é ótima.
qp = QuadraticProgram()
qp.binary_var('x0')  # rota A (00)
qp.binary_var('x1')  # rota B (01)
qp.binary_var('x2')  # rota C (10)

# Custo linear: 10*x0 + 7*x1 + 15*x2
qp.minimize(linear=[10, 7, 15])

qp.linear_constraint([1, 1, 1], sense='==', rhs=1, name='one_route')

# --- 2) Configurar QAOA com reps=p ---
def run_qaoa(p):
    print(f"\n=== Rodando QAOA com p={p} ===")
    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=200), reps=p)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    print("Solução QAOA:", result)
    print("Custo mínimo encontrado:", result.fval)
    return result

# --- 3) Rodar para p=1,2,3 ---
res_p1 = run_qaoa(1)
res_p2 = run_qaoa(5)
res_p3 = run_qaoa(10)

# --- 4) Visualizar probabilidades das soluções ---
def plot_probs(result, p):
    raw = result.min_eigen_solver_result.eigenstate
    probs = raw.binary_probabilities()
    plt.figure(figsize=(4,3))
    plot_histogram(probs, title=f"Distribuição QAOA (p={p})", ax=plt.gca())
    plt.show()

plot_probs(res_p1, 1)
plot_probs(res_p2, 2)
plot_probs(res_p3, 3)
