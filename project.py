"""
Date: 09/12/2024
Auteurs: Célestin Pemmers (matricule: 000562451), Ashe Vazquez Nunez (matricule: 000577005)
Description: Code pour le projet du cours INFO-F302 - Calcul du nombre d'Alcuin d'un graphe.
"""

import networkx as nx
from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool
from copy import copy
from utils import INFINITY


def k_subsets(vec, k: int, choices: set = None, subsets: list = None) -> list:
    """
    Génère tous les sous-ensembles de k éléments d'un ensemble

    :param vec: ensemble considéré
    :param int k: nombre d'éléments dans chaque sous-ensemble
    :param set choices: sous-ensemble en cours de génération, defaults to None
    :param list subsets: liste des sous-ensembles de k éléments, defaults to None
    :return list: liste des sous-ensembles de k éléments
    """
    if choices is None:
        choices = set()
    if k == 0:
        if subsets is None:
            subsets = []
        if choices not in subsets:
            subsets.append(copy(choices))
    else:
        for element in vec:
            if element not in choices:
                choices.add(element)
                subsets = k_subsets(vec, k-1, choices, subsets)
                choices.discard(element)
    return subsets

# = = = = = = = = = = = =
#          Q2
# = = = = = = = = = = = =

def generate_Q2_cnf(G: nx.Graph, k: int) -> tuple[CNFPlus, IDPool]:
    """
    Génère la formule en FNC et l'instance de IDPool pour les questions 2 et 5

    :param nx.Graph G: graphe considéré
    :param int k: nombre de places sur le bateau
    :return tuple[CNFPLUS, IDPool]: formule et instance de IDPool
    """

    n = G.number_of_nodes()
    cnf = CNFPlus()  # formule en FNC associée au problème
    vpool = IDPool()
    
    # 1) Condition de départ
    for vertex in G.nodes:
        cnf.append([-vpool.id((0, vertex))])

    # 2) Condition d'arrêt
    for vertex in G.nodes:
        cnf.append([vpool.id((2*n+1, vertex))])
    
    # 3) Pas conflits sur la rive opposée au berger
    for i in range(2*n+2):
        for edge in G.edges:
            cnf.append([(-1)**(i%2 + 1) * vpool.id((i, edge[0])), (-1)**(i%2 + 1) * vpool.id((i, edge[1]))])
    
    # 4) Les sommets ne peuvent bouger qu'avec le berger
    for i in range(2*n+1):
        for vertex in G.nodes:
            cnf.append([(-1)**(i%2 + 1) * vpool.id((i, vertex)), (-1)**(i%2) * vpool.id((i+1, vertex))])
    
    # 5) Au plus k sommets transportés
    # Cette condition n'a de sens que si k < n. En effet, si k >= n, le bateau condiendra toujours au plus n sommets,
    # donc au plus k
    if k < n:
        subsets = k_subsets(G.nodes,k+1)  # comme k < n, nous avons soit k+1 < n, soit k+1 = n, donc appeler k_subsets fait sens
        for i in range(2*n+1):
            for sub in subsets:
                    clause = []
                    for item in sub:
                        clause.append((-1)**(i%2) * vpool.id((i, item)))
                        clause.append((-1)**(i%2 + 1) * vpool.id((i+1, item)))
                    cnf.append(clause)
    
    return cnf, vpool

def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    """
    Implémentation de la question 2

    :param nx.Graph G: graphe considéré
    :param int k: nombre de places sur le bateau
    :return list[tuple[int, set, set]]: liste de triplets (b, S0, S1)
    """
    n = G.number_of_nodes()
    cnf, vpool = generate_Q2_cnf(G, k)  # génération de la formule en FNC
    
    solver = Minicard(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    if solver.solve():
        res = [[i%2, set(), set()] for i in range(2*n+2)]  # liste des triplets (b, S0, S1)
        for value in solver.get_model():  # fonction de valuation obtenue pour la formule
            take = 2 if value > 0 else 1
            variable = vpool.obj(abs(value))  # variable associée à l'entier abs(value)
            # Si value > 0, variable évaluée à True. Donc, à la configuration i, le sommet a est sur la rive 1 (troisième élément de (b, S0, S1)).
            # Si value < 0, variable évaluée à False. Donc, à la configuration i, le sommet a est sur la rive 0 (deuxième élément de (b, S0, S1)).
            res[variable[0]][take].add(variable[1])
    else:
        res = None
    
    return res

# = = = = = = = = = = = =
#          Q3
# = = = = = = = = = = = =

def find_alcuin_number(G: nx.Graph, lower: int = 1, upper: int = None) -> int:
    """
    Implémentation de la question 3: trouver le nombre d'Alcuin d'un graphe

    :param nx.Graph G: graphe considéré
    :param int lower: borne inférieure de la recherche binaire, defaults to 1
    :param int upper: borne supérieure de la recherche binaire, defaults to None
    :return int: nombre d'Alcuin de G
    """
    n = G.number_of_nodes()
    if upper is None:
        upper = n
    
    if lower == upper:  # cas de base
        return lower

    k = (upper+lower)//2
    if gen_solution(G,k) is not None:  # le nombre d'Alcuin est au plus k
        return find_alcuin_number(G, lower, k)
    else:  # le nombre d'Alcuin est au moins k+1
        return find_alcuin_number(G, k+1, upper)

# = = = = = = = = = = = =
#          Q5
# = = = = = = = = = = = =

def generate_Q5_cnf(G: nx.Graph, k: int, c: int) -> tuple[CNFPlus, IDPool]:
    """
    Génère le formule en FNC et l'instance de IDPool pour la question 5

    :param nx.Graph G: graphe considéré
    :param int k: nombre de places sur le bateau
    :param int c: nombre de compartiments
    :return tuple[CNFPlus, IDPool]: formule et instance de IDPool
    """
    n = G.number_of_nodes()
    cnf, vpool = generate_Q2_cnf(G, k)  # contraintes héritées de la question 2

    # 1) Tout sommet est dans au plus un compartiment
    for i in range(1, 2*n+2):
        for vertex in G.nodes:
            for j in range(c+1):
                for j_prime in range(j):
                    cnf.append([-vpool.id((i, vertex, j)), -vpool.id((i, vertex, j_prime))])

    # 2) Tout sommet est dans au moins un compartiment
    for i in range(1, 2*n+2):
        for vertex in G.nodes:
            cnf.append([vpool.id((i, vertex, j)) for j in range(c+1)])

    # 3) Un sommet n'était dans aucun compartiment non nul <=> il n'a pas été bougé
    for i in range(2*n+1):
        for vertex in G.nodes:
            
            # => : le sommet n'est dans aucune compartiment non nul
            cnf.append([-vpool.id((i+1, vertex, 0)), (-1)**(i%2) * vpool.id((i, vertex)), (-1)**(i%2 + 1) * vpool.id((i+1, vertex))])
            
            # <= : le sommet n'a pas bougé
            cnf.append([(-1)**(i%2+1) * vpool.id((i, vertex)), vpool.id((i+1, vertex, 0))])
            cnf.append([(-1)**(i%2) * vpool.id((i+1, vertex)), vpool.id((i+1, vertex, 0))])

    # 4) Deux objets en conflit ne sont jamais dans le même compartiment non nul
    for i in range(1, 2*n+2):
        for edge in G.edges:
            for j in range(1, c+1):
                cnf.append([-vpool.id((i, edge[0], j)), -vpool.id((i, edge[1], j))])
    
    return cnf, vpool


def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    """
    Implémentation de la question 5

    :param nx.Graph G: graphe considéré
    :param int k: nombre de places sur le bateau
    :param int c: nombre de compartiments
    :return list[tuple[int, set, set, tuple[set]]]: liste de 4-uples (b, S0, S1, (X1, ..., Xc))
    """
    n = G.number_of_nodes()
    cnf, vpool = generate_Q5_cnf(G, k, c)  # génération de la formule en FNC

    solver = Minicard(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    if solver.solve():
        res = []  # liste des 4-uples (b, S0, S1, (X1, ..., Xc))
        for i in range(2*n+2):
            banks = [i%2, set(), set()]  # triplet (b, S0, S1)
            compartments = [set() for _ in range(c)]  # compartiments du bateau (c-uple (X1, ..., Xc))
            banks.append(compartments)
            res.append(banks)
        
        for value in solver.get_model():  # fonction de valuation obtenue pour la formule
            variable = vpool.obj(abs(value))  # variable associée à l'entier abs(value)
            if len(variable) == 2:  # variable x_{configuration, sommet}
                take = 2 if value > 0 else 1
                # Si value > 0, variable évaluée à True. Donc, à la configuration i, le sommet a est sur la rive 1 (troisième élément de (b, S0, S1)).
                # Si value < 0, variable évaluée à False. Donc, à la configuration i, le sommet a est sur la rive 0 (deuxième élément de (b, S0, S1)).
                res[variable[0]][take].add(variable[1])
            else:  # variable p_{configuration, sommet, compartiment}
                # si la variable ci-dessus est évaluée à True, le sommet est dans le compartiment indiqué
                # s'il est dans le compartiment 0, il n'est pas transporté => on veut compartiment != 0
                if value > 0 and variable[2] != 0:
                    # on accède à l'élément (X1_i, ..., Xc_i) du 4-uplet (b_i, S0_i, S1_i, (X1_i, ..., Xc_i)) (indice 3),
                    # et sélectionne le compartiment numéro variable[2] (indice variable[2]-1)
                    res[variable[0]][3][variable[2]-1].add(variable[1])
    else:
        res = None
    
    return res

# = = = = = = = = = = = =
#          Q6
# = = = = = = = = = = = =

def find_c_alcuin_number(G: nx.Graph, c: int, lower: int = 1, upper: int = None) -> int:
    """
    Implémentation de la question 6: trouver le nombre de c-Alcuin d'un graphe

    :param nx.Graph G: graphe en question
    :param int c: nombre de compartiments
    :param int lower: borne inférieure de la recherche binaire, defaults to 1
    :param int upper: borne supérieure de la recherche binaire, defaults to None
    :return int: nombre de c-Alcuin du graphe
    """
    n = G.number_of_nodes()
    if upper is None:
        upper = n
        # on retourne INFINITY s'il n'existe aucune une séquence c-valide
        if gen_solution_cvalid(G, n, c) is None:
            return INFINITY 

    if lower >= upper:  # cas de base 
        return lower

    k = (upper+lower)//2
    if gen_solution_cvalid(G, k, c) is not None:  # le nombre de c-Alcuin est au plus k
        return find_c_alcuin_number(G, c, lower, k)
    else:  # nombre de c-Alcuin est au moins k+1
        return find_c_alcuin_number(G, c, k+1, upper)