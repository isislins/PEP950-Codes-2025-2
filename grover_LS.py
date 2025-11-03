import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np
import math

# -------- dicionário --------
result_dict = {
    "010000001000100000": [0.978767128,183],
    "000010000001100000": [0.971196480,141],
    "001001000010001000": [0.975445962,178],
    "000010010001001001": [0.971206461,175],
    "000010000010001001": [0.971209683,162],
    "001001000001011001": [0.975464729,171],
    "000010001001011000": [0.971221206,158],
    "000010000010000011": [0.971227445,176],
    "000010001010000001": [0.971230511,179],
    "000010010001010000": [0.971238632,175],
    "000010000010010000": [0.971240449,162],
    "001001000001100000": [0.975529784,171],
    "000010000010001010": [0.971250465,176],
    "000010001010001000": [0.971252194,179],
    "000010000011000000": [0.971261336,183],
    "000010000010010001": [0.971263448,176],
    "000010000010011000": [0.971270768,176]
}

def funcao_conf(a,b,c,d):
    return 0

# -------- parâmetros --------
SEED = "000010000010011000"  # seed inicial (soma = 7 neste caso)
BLOCK_SIZE = 3               # 3 bits por equipamento
MAX_TOTAL = 7                # soma máxima global
N = len(SEED)
MAX_ITER = 50
MAX_NO_IMPROVE = 20

# -------- conversões utilitárias --------
def binary_to_blocks(bin_str, block_size=BLOCK_SIZE):
    return [bin_str[i:i+block_size] for i in range(0, len(bin_str), block_size)]

def blocks_to_binary(blocks):
    return "".join(blocks)

def blocks_to_ints(blocks):
    return [int(b, 2) for b in blocks]

def ints_to_blocks(ints, block_size=BLOCK_SIZE):
    return [format(v, f"0{block_size}b") for v in ints]

def ints_to_binary(ints, block_size=BLOCK_SIZE):
    return blocks_to_binary(ints_to_blocks(ints, block_size))

def total_units_from_binary(bin_str):
    return sum(blocks_to_ints(binary_to_blocks(bin_str)))

def truncate(number, decimals=0):
    if decimals < 0:
        raise ValueError("Decimal places must be non-negative")
    factor = 10 ** decimals
    return math.trunc(number * factor) / factor

# -------- função de avaliação (usa result_dict) --------
def evaluate_seed(seed, current):
    """Retorna score (quanto maior melhor). Seeds fora do dicionário penalizadas com 0."""
    try:
        val_max, val_min = result_dict[seed]
        max_val_max, max_val_min = result_dict[current]
        if val_max > max_val_max:
            best = seed
            print(f"  [Melhora] {current} ({max_val_max}, {max_val_min}) -> {seed} ({val_max}, {val_min})")
            improvement = True
        else:
            best = current
            improvement = False
            print(f"  [Sem melhora] {current} ({max_val_max}, {max_val_min}) -> {seed} ({val_max}, {val_min})")
    except KeyError:
        best = current  # seed não encontrada no dicionário
    return best, improvement

# -------- mutação controlada com restrição global --------
def mutate_with_global_limit(seed,
                             max_total=MAX_TOTAL,
                             block_size=BLOCK_SIZE,
                             require_in_dict=True,
                             retries=50):
    """
    Gera uma mutação que respeita a soma total <= max_total.
    Operadores (probabilísticos):
      - mover 1 unidade de um equipamento para outro (soma inalterada)
      - incrementar (soma+1) somente se total < max_total
      - decrementar (soma-1)
    Se require_in_dict=True, tenta gerar uma seed presente em result_dict.
    """
    blocks = binary_to_blocks(seed, block_size)
    vals = blocks_to_ints(blocks)
    num_blocks = len(vals)
    total = sum(vals)

    for _ in range(retries):
        new_vals = vals.copy()
        op = random.random()
        # 60% move, 20% increment (se possível), 20% decrement (se possível)
        if op < 0.6:
            # mover 1 unidade: escolhe doador (valor>0) e receptor (valor<7)
            donors = [i for i, v in enumerate(new_vals) if v > 0]
            receivers = [j for j, v in enumerate(new_vals) if v < 7]
            # precisa existir donor e receiver != donor
            if not donors or len(receivers) <= 1:
                # não é possível mover com variedade; tente outro operador
                continue
            i = random.choice(donors)
            j_candidates = [r for r in receivers if r != i]
            if not j_candidates:
                continue
            j = random.choice(j_candidates)
            new_vals[i] -= 1
            new_vals[j] += 1
        elif op < 0.8:
            # incrementar um equipamento (só se total < max_total)
            if total < max_total:
                candidates = [i for i, v in enumerate(new_vals) if v < 7]
                if not candidates:
                    continue
                i = random.choice(candidates)
                new_vals[i] += 1
            else:
                continue
        else:
            # decrementar um equipamento (se houver algum > 0)
            donors = [i for i, v in enumerate(new_vals) if v > 0]
            if not donors:
                continue
            i = random.choice(donors)
            new_vals[i] -= 1

        # validação final
        if sum(new_vals) <= max_total and all(0 <= x <= 7 for x in new_vals):
            candidate = ints_to_binary(new_vals, block_size)
            if require_in_dict and str(candidate) not in result_dict:
                # se exigir presença no dicionário, tentar novamente
                continue
            return candidate

    # fallback: retorna a própria seed se não encontrou mutação válida
    return seed

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

    for i in range(int(np.pi/4 * np.sqrt(2**n))):
        # --- 2. Oráculo (marca o estado alvo) ---

        ### Verifica um novo conjunto resposta -> Checo se o novo alvo é melhor que o atual
        X = funcao_conf(estado=target) 
        ### Novo alvo for melhor que o atual -> Oraculo atualiza para novo alvo
        ### Caso contrario, mantem o oraculo no alvo atual
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

# -------- loop iterativo usando a mutação controlada --------
def grover_iterative_global_limit(seed,
                                  max_iter=MAX_ITER,
                                  max_no_improve=MAX_NO_IMPROVE,
                                  require_in_dict=True):
    # valida seed inicial
    if total_units_from_binary(seed) > MAX_TOTAL:
        raise ValueError("Seed inicial viola o limite total de equipamentos.")

    current = seed
    no_improve = 0

    print(f"[Início] Seed inicial: {seed} (total={total_units_from_binary(seed)})")

    for it in range(1, max_iter + 1):
        # gerar mutação que respeita o limite global
        mutated = mutate_with_global_limit(current,
                                           max_total=MAX_TOTAL,
                                           require_in_dict=require_in_dict,
                                           retries=200)
        total_mut = total_units_from_binary(mutated)
        print(f"\nIter {it}: candidato gerado: {mutated} (total={total_mut})")
        qc, counts = grover_search(mutated)
        aux = 0
        for i in counts:
            # Inverta a string para comparar com o target
            bitstring = i[::-1]
            if aux == 0:
                found = [bitstring, counts[i]]
                aux = 1
            else:
                if counts[i] > found[1]:
                    found = [bitstring, counts[i]]
        print("Grover found:", found[0])
        current, improvement = evaluate_seed(found[0], current)
        if improvement:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            print("\n🔚 Sem melhora após várias tentativas — parando.")
            break

    print(f"\n🏁 Melhor seed final: {current} (Confiabilidade={result_dict[current][0]}) Custo = {result_dict[current][0]}")
    return current
    

       

# -------- exemplo de execução --------
if __name__ == "__main__":
    grover_iterative_global_limit(SEED,
                                                           max_iter=MAX_ITER,
                                                           max_no_improve=MAX_NO_IMPROVE,
                                                           require_in_dict=True)
