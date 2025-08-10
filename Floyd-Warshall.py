# Universidade Federal de Pernambuco
# Centro de Tecnologia e Geociencias
# Departamento de Engenharia de Producao
# Curso de Pos-Graduacao em Engenharia de Producao
# Recife, 22 de setembro de 2023

# Disciplina: Otimizacao Combinatoria (PEP950)
# Professora: Isis Didier Lins

# Uma implementacao do algoritmo de Floyd-Warshall para encontrar o caminho minimo
# entre um no fonte e todos os demais nos do grafo.

import numpy as np

# Entrada do grafo orientado:
V = np.arange(6) # Conjunto de nos (array).
n = len(V) # Numero de n�s no grafo.
M = np.inf # Valor infinito.
A = np.array([[M,6,4,M,M,M],
              [M,M,2,2,M,M],
              [M,M,M,1,2,M],
              [M,M,M,M,M,7],
              [M,M,M,1,M,3],
              [M,M,M,M,M,M]])# Matriz de adjacencias com pesos.
# A = np.array([[M,10,15,M,M],
#                 [M,M,25,0,5],
#                 [M,-20,M,M,M],
#                 [M,M,M,M,-5],
#                 [M,M,30,10,M]])
# A = np.array([[M,10,15,M,M,M],
#               [M,M,25,0,5,-2],
#               [M,-20,M,M,M,M],
#               [M,M,M,M,-5,M],
#               [M,M,30,10,M,M],
#               [M,M,M,-2,M,M]])
# A = np.array([[M,10,15,M,M,M],
#               [M,M,25,0,5,-5],
#               [M,-20,M,M,M,M],
#               [M,M,M,M,-5,M],
#               [M,M,30,10,M,M],
#               [M,M,M,-5,M,M]])

# A = np.array([[M,5,M,10,M,M,M,M,M,M,M,M],
#               [M,M,7,M,1,M,M,M,M,M,M,M],
#               [M,M,M,M,M,4,M,M,M,M,M,M],
#               [M,M,M,M,3,M,11,M,M,M,M,M],
#               [M,M,M,M,M,3,M,7,M,M,M,M],
#               [M,M,M,M,M,M,M,M,5,M,M,M],
#               [M,M,M,M,M,M,M,2,M,9,M,M],
#               [M,M,M,M,M,M,M,M,0,M,1,M],
#               [M,M,M,M,M,M,M,M,M,M,M,12],
#               [M,M,M,M,M,M,M,M,M,M,2,M],
#               [M,M,M,M,M,M,M,M,M,M,M,4],
#               [M,M,M,M,M,M,M,M,M,M,M,M]])# Matriz de adjacencias com pesos.


# Inicializacao da matriz de predecessores de cada par de nos:
pred = np.ones((n,n),dtype=np.int16) * -1# Todos os predecessores iniciam com 0 (no inexistente no grafo 
               # original).

# Inicializa��o da matriz de dist�ncias:
# dist = np.ones((n,n)) * M # Todas as distancias iniciam com valor infinito.
dist = A.copy() # Para criar uma copia, c.c. mudancas em dist alteram A tambem.
for i in range(n):
    for j in range(n):
        if dist[i,j] != M:
            pred[i,j] = i
np.fill_diagonal(dist,0) # A distancia de um no para ele mesmo e zero, entao a diagonal principal da matriz de distancia e 0. 

for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i,j] > dist[i,k] + dist[k,j]:
                dist[i,j] = dist[i,k] + dist[k,j]
                pred[i,j] = pred[k,j]

def findPath(i,j,pred):
    path = []
    k = pred[i,j]
    path.insert(0,k)
 #   print(i,j,k)
    while k != -1 and k != i:
        k = pred[i,k]
        path.insert(0,k)
    if k != -1:
        path.append(j)    
#    print(i,j,k)
    return path

            
# Impressao de resultados:
print('Distancia minima entre cada par de nos:\n',dist)

flag = False
for i in range(n):
    if dist[i,i] != 0:
        flag = True
        
if flag == False:        
    print('Matriz de predecessores de cada par de nos:\n',pred)
    paths = []
    for i in range(n):
        path = []
        for j in range(n):
            path.append(findPath(i,j,pred))
            if path[j][0] != -1:
                print(f'Caminho minimo partindo de {i} para {j}: {path[j]}. Distancia: {dist[i,j]}')
        paths.append(path)
    # print('Caminhos minimos entre cada par de nos:\n',paths)
else:
    print('\nHa ciclo negativo no grafo!\n')