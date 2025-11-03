import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate 
from qiskit_algorithms import Grover, AmplificationProblem
from qiskit_aer import AerSimulator

# --- 1. PREPARAÇÃO (CLÁSSICA) ---

def encode_array(arr):
    """Codifica um array de 6 números em uma string binária de 18 bits."""
    if len(arr) != 6:
        raise ValueError("O array deve ter exatamente 6 elementos.")
    return "".join([f"{x:03b}" for x in arr])

# Transcrição dos dados (Array -> Confiabilidade)
data_pairs = [  ## os 2 primeiros equipamentos alocados em 1 subsistema com limite de 2; os 4 ultimos em outro subsistema com limite de 5
    ([0, 2, 0, 0, 0, 5], 0.958288127),
    ([1, 1, 0, 0, 0, 5], 0.962563836),
    ([0, 2, 0, 0, 1, 4], 0.963953628),
    ([0, 2, 1, 0, 0, 4], 0.966116516),
    ([2, 0, 0, 0, 0, 5], 0.966194361),
    ([0, 2, 0, 0, 2, 3], 0.967148559),
    ([1, 1, 0, 0, 1, 4], 0.968254615),
    ([0, 2, 1, 0, 1, 3], 0.968368272),
    ([0, 2, 0, 0, 3, 2], 0.968950270),
    ([0, 2, 2, 0, 0, 3], 0.969227913),
    ([0, 2, 1, 0, 2, 2], 0.969638099),
    ([0, 2, 0, 0, 4, 1], 0.969966304),
    ([0, 2, 2, 0, 1, 2], 0.970122874),
    ([0, 2, 1, 0, 3, 1], 0.970354190),
    ([1, 1, 1, 0, 0, 4], 0.970427154),
    ([0, 2, 0, 1, 0, 4], 0.970452065),
    ([0, 2, 3, 0, 0, 2], 0.970464539),
    ([0, 2, 0, 0, 5, 0], 0.970539273),
    ([0, 2, 2, 0, 2, 1], 0.970627568),
    ([0, 2, 1, 0, 4, 0], 0.970758013),
    ([0, 2, 0, 1, 1, 3], 0.970813207),
    ([0, 2, 3, 0, 1, 1], 0.970820242),
    ([0, 2, 2, 0, 3, 0], 0.970912178),
    ([0, 2, 1, 1, 0, 3], 0.970951079),
    ([0, 2, 4, 0, 0, 1], 0.970956037),
    ([0, 2, 0, 1, 2, 2], 0.971016866),
    ([0, 2, 3, 0, 2, 0], 0.971020833),
    ([0, 2, 1, 1, 1, 2], 0.971094615),
    ([0, 2, 4, 0, 1, 0], 0.971097411),
    ([0, 2, 0, 1, 3, 1], 0.971131714),
    ([0, 2, 2, 1, 0, 2], 0.971149412),
    ([0, 2, 1, 1, 2, 1], 0.971175559),
    ([0, 2, 0, 1, 4, 0], 0.971196480),
    ([0, 2, 2, 1, 1, 1], 0.971206461),
    ([0, 2, 1, 1, 3, 0], 0.971221206),
    ([0, 2, 0, 2, 0, 3], 0.971227445),
    ([0, 2, 2, 1, 2, 0], 0.971238632),
    ([0, 2, 0, 2, 1, 2], 0.971250466),
    ([0, 2, 0, 2, 2, 1], 0.971263448),
    ([0, 2, 0, 2, 3, 0], 0.971270768),
    ([1, 1, 0, 0, 2, 3], 0.971463802),
    ([2, 0, 0, 0, 1, 4], 0.971906604),
    ([1, 1, 1, 0, 1, 3], 0.972688957),
    ([1, 1, 0, 0, 3, 2], 0.973273551),
    ([1, 1, 2, 0, 0, 3], 0.973552434),
    ([1, 1, 1, 0, 2, 2], 0.973964450),
    ([2, 0, 1, 0, 0, 4], 0.974087337),
    ([1, 1, 0, 0, 4, 1], 0.974294119),
    ([1, 1, 2, 0, 1, 2], 0.974451388),
    ([1, 1, 1, 0, 3, 1], 0.974683735),
    ([1, 1, 0, 1, 0, 4], 0.974782047),
    ([1, 1, 0, 0, 5, 0], 0.974869645),
    ([1, 1, 2, 0, 2, 1], 0.974958333),
    ([1, 1, 1, 0, 4, 0], 0.975089360),
    ([2, 0, 0, 0, 2, 3], 0.975127895),
    ([1, 1, 0, 1, 1, 3], 0.975144801),
    ([1, 1, 2, 0, 3, 0], 0.975244214),
    ([1, 1, 0, 1, 2, 2], 0.975349368),
    ([1, 1, 0, 1, 3, 1], 0.975464729),
    ([1, 1, 0, 1, 4, 0], 0.975529784),
    ([2, 0, 1, 0, 1, 3], 0.976357671),
    ([2, 0, 0, 0, 3, 2], 0.976944470),
    ([2, 0, 1, 0, 2, 2], 0.977637974),
    ([2, 0, 0, 0, 4, 1], 0.977968887),
    ([2, 0, 1, 0, 3, 1], 0.978359973),
    ([2, 0, 0, 0, 5, 0], 0.978546584),
    ([2, 0, 1, 0, 4, 0], 0.978767128)
]

# Cria o "banco de dados" de soluções conhecidas (String Binária -> Confiabilidade)
known_solutions = {encode_array(arr): val for arr, val in data_pairs}
n_bits = 18 # 6 números * 3 bits/número

# Solucao inicial (seu ótimo local)
incumbent_solution = "000010000010011000"
incumbent_reliability = 0.971270768

# Agendamento estático BBW2 do artigo
bbw2_static_schedule = [
    0, 0, 0, 1, 1, 0, 1, 1, 2, 1, 2, 3, 1, 4, 5, 1, 6, 2, 7, 9, 
    11, 13, 16, 5, 20, 24, 28, 34, 2, 41, 49, 4, 60, 72, 9, 88, 
    105, 125, 3, 149, 22, 183
]

# Inicializa o simulador Qiskit
#simulator = AerSimulator()
simulator = AerSimulator(method='statevector', 
                        max_parallel_experiments=1,
                        max_parallel_shots=1)

print("--- INICIANDO BUSCA LOCAL HÍBRIDA (QGO + BBW2) ---")
print(f"Buscando ótimo global a partir de:\n  {incumbent_solution} (Conf: {incumbent_reliability})\n")
print(f"Total de soluções conhecidas: {len(known_solutions)}")
print(f"Tamanho do agendamento BBW2: {len(bbw2_static_schedule)} iterações\n")
print("-" * 50)

# --- 2. LOOP DE OTIMIZAÇÃO (HÍBRIDO) ---
marked_states = []
# Itera sobre o agendamento estático
for c, r in enumerate(bbw2_static_schedule):
    
    # Passo 2b (Clássico): Encontrar todos os estados melhores que o incumbente
    marked_states = [
        state for state, reliability in known_solutions.items() 
        #if reliability > incumbent_reliability
    ]

    if marked_states == None or len(marked_states) == 0:
        print(f"\n--- Fim da busca na iteração {c+1} ---")
        print("Nenhuma solução conhecida é melhor que o incumbente atual.")
        print("Ótimo global do conjunto encontrado!")
        break
        
    print(f"Iteração {c+1}/{len(bbw2_static_schedule)} (Usando r={r} rotações)")
    print(f"  Buscando soluções com conf. > {incumbent_reliability:.9f}")
    # print(f"  Estados marcados (M_c): {marked_states}") # Descomente para depurar

    # Passo 2c (Quântico): Construir e rodar Grover
    
    # 1. Cria um vetor de fases de tamanho 2^n_bits, preenchido com 1s
    diag = np.ones(2**n_bits)
    # 2. Para cada estado marcado, mude a fase para -1 no índice correspondente
    for state_str in marked_states:
        idx = int(state_str, 2) # Converte a string binária para inteiro
        diag[idx] = -1
    # 3. Cria o oráculo a partir do vetor de fases
    oracle = DiagonalGate(diag) # <-- MUDANÇA 3: Usando DiagonalGate
    
    
    # Estado inicial: superposição uniforme sobre todos os n_bits
    init_state = QuantumCircuit(n_bits)
    init_state.h(range(n_bits))
    
    # Definimos o problema de amplificação
    problem = AmplificationProblem(
        oracle=oracle,
        state_preparation=init_state,
        is_good_state=marked_states # Define quais são os estados de "sucesso"
    )
    
    # Criamos o circuito de Grover com 'r' rotações
    grover_circuit = Grover(iterations=r).construct_circuit(problem)
    grover_circuit.measure_all()
    
    # --- INÍCIO DAS MUDANÇAS 4 e 5 ---
    # "Desdobra" o circuito de alto nível em portas básicas que o simulador entende
    transpiled_circuit = transpile(grover_circuit, simulator)
    
    # Executamos o circuito transpilado no simulador
    result = simulator.run(transpiled_circuit, shots=1, memory=True).result()

    
    medida = result.get_memory()[0]
    
    # Passo 2d (Clássico): Avaliar o resultado
    if medida in marked_states:
        if known_solutions[medida] >= incumbent_reliability:
            # SUCESSO!
            novo_incumbente = medida
            nova_confiabilidade = known_solutions[novo_incumbente]
            
            print(f"  SUCESSO! Grover encontrou um estado melhor:")
            print(f"  -> {novo_incumbente} (Conf: {nova_confiabilidade:.9f})\n")
            
            # Atualiza o incumbente para a próxima iteração
            incumbent_solution = novo_incumbente
            incumbent_reliability = nova_confiabilidade
        else:
            # FALHA
            print(f"  Solução possível obtida: Medição ({medida}), porém não é um estado ótimo.\n")
        known_solutions.pop(medida, None)  # Remove a solução medida do conjunto       

print("\n" + "=" * 50)
print("--- BUSCA CONCLUÍDA ---")
print(f"Melhor solução final encontrada no conjunto:")
print(f"  {incumbent_solution}")
print(f"Confiabilidade final: {incumbent_reliability:.9f}")
print("=" * 50)