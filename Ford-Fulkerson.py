# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:58:16 2023

@author: isisl
"""

# Universidade Federal de Pernambuco
# Centro de Tecnologia e Geociencias
# Departamento de Engenharia de Producao
# Curso de Pos-Graduacao em Engenharia de Producao
# Recife, 29 de setembro de 2023

# Disciplina: Otimizacao Combinatoria (PEP950)
# Professora: Isis Didier Lins

# Uma implementacao do algoritmo de Ford-Fulkerson para encontrar o fluxo 
# maximo numa rede.

import numpy as np

# Entrada do grafo orientado:
V = np.arange(7) # Conjunto de nos (array).
n = len(V) # Numero de nós no grafo.
M = np.inf # Valor infinito.

# Matriz de adjacencias com pesos:
A = np.array([[0,3,5,7,0,0,0],
              [0,0,0,0,2,0,0],
              [0,0,0,10,5,0,0],
              [0,0,0,0,0,15,0],
              [0,0,0,0,0,0,5],
              [0,0,0,0,0,0,15],
              [0,0,0,0,0,0,0]]) 

A_res = A.copy()

def bfs(A_res,s):
    edges = []
    
    #marcar todos os vértices como não visitados
    visited = [False] * (len(A_res))

    #criando uma fila vazia para o BFS
    queue = []

    #vai para o nó de origem, marca como vistado e o insere na fila
    queue.append(s)
    visited[s] = True
        #enquando a fila não for visitada
    while queue:
        #retira o último verso inserido na fila e o imprime
        s = queue.pop(0)
       # print(s, " ") #respostas que formam o BFS

        #obter todos os vértices adjacentes dos vértices desenfileirados (vértices para onde consigo ir)
        for j in range(len(A_res[s])):
            #print(visited[i])
            if visited[j] == False and A_res[s,j] != 0:
                queue.append(j)
                visited[j] = True
                edges.append([s,j])
    
    return edges 

def findPath(A_res,s,t):
    s = int(s)
    t = int(t)
    path = []
    edges = bfs(A_res,s)
    initial = t
    while initial != s:
        for edge in edges:
            if edge[1] == initial:
                path.insert(0,edge)
                initial = edge[0]
                break # Para o for (encontrou aresta com no final igual a t)
        else:
            break # Para o while (nao encontrou aresta com no final igual a t, ou seja, nao tem caminho ate t) 
    return path

flag = True
maxFlow = 0
s = 0
t = 6

while flag:
    path = findPath(A_res,s,t)
    if not path:
        break
    delta = []
    for edge in path:
        delta.append(A_res[edge[0],edge[1]])
    delta = min(delta)
    maxFlow += delta
    for edge in path:
        # Atualizacao do fluxo se [edge[1],edge[0]] nao pertence a rede
        # original; atualização da capacidade residual se [edge[1],edge[0]] 
        # pertence a rede original:
        A_res[edge[1],edge[0]] += delta 
        # Atualizacao da capacidade residual se [edge[0],edge[1]] pertence a
        # rede original; atualização do flluxo se [edge[0],edge[1]] nao  
        # pertence a rede original:
        A_res[edge[0],edge[1]] -= delta # Atualizacao da capacidade residual
        
print(f'Fluxo maximo na rede partindo de {s} ate {t}: {maxFlow}.')
print(A_res)