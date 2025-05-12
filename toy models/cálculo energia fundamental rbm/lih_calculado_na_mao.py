import numpy as np
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator
from scipy.sparse.linalg import eigsh
from itertools import combinations

# 1. Carrega os dados
data = np.load("lih_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])     # 6
nelec = int(data["nelec"])     # 4

# 2. Constrói Hamiltoniano em segunda quantização
H = FermionOperator()

# Termos de 1 elétron
for p in range(n_orb):
    for q in range(n_orb):
        coef = h1[p, q]
        if abs(coef) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", coef)

# Termos de 2 elétrons
for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                coef = 0.5 * eri[p, q, r, s]
                if abs(coef) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", coef)

# 3. Jordan-Wigner
H_jw = jordan_wigner(H)

# 4. Gera todas as 15 configurações (ocupações binárias com 4 elétrons em 6 orbitais)
def generate_configurations(n, k):
    configs = []
    for occ in combinations(range(n), k):
        state = np.zeros(n, dtype=int)
        state[list(occ)] = 1
        configs.append(state)
    return np.array(configs)

configs = generate_configurations(n_orb, nelec)

# 5. Avalia H em cada par de configurações (15x15 matriz)
psi_vals = [1.0] * len(configs)  # assume função de onda igual para projeção
H_matrix = np.zeros((len(configs), len(configs)), dtype=complex)

for i, sigma_i in enumerate(configs):
    for j, sigma_j in enumerate(configs):
        contrib = 0
        for term, coeff in H_jw.terms.items():
            s_i = sigma_i.copy()
            s_j = sigma_j.copy()
            phase = 1.0 + 0j
            for qubit, op in term:
                if op == 'Z':
                    phase *= (-1) ** s_j[qubit]
                elif op in ('X', 'Y'):
                    s_j[qubit] = 1 - s_j[qubit]
                    if op == 'Y':
                        phase *= 1j if sigma_j[qubit] == 0 else -1j
            if np.array_equal(s_j, sigma_i):
                contrib += coeff * phase
        H_matrix[i, j] = contrib

# 6. Diagonaliza
eigvals, eigvecs = np.linalg.eigh(H_matrix.real)

# 7. Resultado
E_ground = eigvals[0]
print(f"⚛️ Energia eletrônica exata (FCI) no subespaço de 15 estados: {E_ground:.6f} Ha")