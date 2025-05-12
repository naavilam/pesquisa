from openfermion import FermionOperator, jordan_wigner, hermitian_conjugated, load_operator
import numpy as np

# Carrega os dados do PySCF
data = np.load("h2_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])

# Cria Hamiltoniano em segunda quantização
H = FermionOperator()

# Termos de um elétron
for p in range(n_orb):
    for q in range(n_orb):
        coef = h1[p, q]
        if abs(coef) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", coef)

# Termos de dois elétrons
for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                coef = 0.5 * eri[p, q, r, s]
                if abs(coef) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", coef)

# Aplica Jordan-Wigner
H_jw = jordan_wigner(H)

# Mostra os 10 primeiros termos do Hamiltoniano mapeado
print("Hamiltoniano mapeado (Jordan–Wigner):")
for term in list(H_jw.terms.items())[:10]:
    print(term)