from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.visualization import plot_histogram

def grover_search(target: str, shots=1024):
    """
    Executa o algoritmo de Grover para buscar um estado alvo específico.
    
    Args:
        target (str): Estado alvo em binário (ex: "11", "01", etc.)
        shots (int): Número de medições no simulador.
    """
    n = len(target)  # número de qubits necessários
    qc = QuantumCircuit(n, n)
    
    # --- 1. Superposição inicial ---
    qc.h(range(n))

    # --- 2. Oráculo (marca o estado alvo) ---
    # Aplicamos X nos qubits que devem ser 0 para inverter a lógica
    for i, bit in enumerate(target):
        if bit == "0":
            qc.x(i)

    # Multi-controlled Z: implementado como H+MCX+H
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)  # porta CCX generalizada
    qc.h(n-1)

    # Reverter os X aplicados
    for i, bit in enumerate(target):
        if bit == "0":
            qc.x(i)

    # --- 3. Difusão (operador de Grover) ---
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))

    # --- 4. Medição ---
    qc.measure(range(n), range(n))

    # --- Simulação ---
    sim = Aer.get_backend('qasm_simulator')
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=1024).result()
    counts = result.get_counts()

    return qc, counts


# Exemplo de uso
target = "01"   # valor que queremos encontrar
qc, counts = grover_search(target)

print("Resultado das medições:", counts)
print("Esperado:", target)

qc.draw("mpl")  # desenha o circuito
plot_histogram(counts).show()
