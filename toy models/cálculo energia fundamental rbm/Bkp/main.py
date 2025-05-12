import numpy as np
from openfermion import FermionOperator, jordan_wigner
from rbm_h2 import RBM, generate_configurations, train_rbm_variacional, plot_energia

# ----------------- Etapa 1: Carregar integrals do PySCF -----------------
data = np.load("h2_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])

# ----------------- Etapa 2: Construir Hamiltoniano em segunda quantização -----------------
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

# ----------------- Etapa 3: Jordan-Wigner -----------------
H_jw = jordan_wigner(H)

# ----------------- Etapa 4: Gerar configurações -----------------
n_qubits = 2 * n_orb
configs = generate_configurations(n_qubits, nelec)

# ----------------- Etapa 5: Inicializar RBM -----------------
rbm = RBM(n_visible=n_qubits, n_hidden=n_qubits)

# ----------------- Etapa 6: Treinamento -----------------
energia_por_epoca = train_rbm_variacional(rbm, configs, H_jw, epochs=100, lr=0.0001)

# ----------------- Etapa 7: Gráfico -----------------
# plot_energia(energia_por_epoca)


import matplotlib.pyplot as plt

def visualizar_psi(rbm, configs):
    psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
    probs = np.abs(psi_vals) ** 2
    probs /= np.sum(probs)

    labels = [''.join(str(int(x)) for x in cfg.astype(int)) for cfg in configs]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, probs)
    plt.xticks(rotation=90)
    plt.ylabel("Probabilidade $|\Psi(\\sigma)|^2$")
    plt.xlabel("Configuração $\\sigma$")
    plt.title("Distribuição da Função de Onda Aprendida pela RBM")
    plt.tight_layout()
    plt.show()


visualizar_psi(rbm, configs)