import numpy as np
from rbm_h2 import RBM, generate_configurations, train_rbm_variacional
from openfermion import FermionOperator, jordan_wigner
import matplotlib.pyplot as plt

# Carrega os integrais do PySCF (STO-3G para H2)
data = np.load("lih_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])
print(nelec)
print(n_orb)

# Cria Hamiltoniano em segunda quantização em termos de operadores fermiônicos
H = FermionOperator()
for p in range(n_orb):
    for q in range(n_orb):
        coef = h1[p, q]
        if abs(coef) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", coef)

for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                coef = 0.5 * eri[p, q, r, s]
                if abs(coef) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", coef)



# Aplica Jordan-Wigner
theta = jordan_wigner(H) # Parametro da RBM


# Gera todas as configurações possíveis para 4 orbitais com 2 elétrons

n_orb = int(data["n_orb"])         # Ex: 6
nelec = int(data["nelec"])         # Ex: 4
n_spin_orb = 2 * n_orb             # Ex: 12
print(n_orb)
print(nelec)
print(n_spin_orb)
configs = generate_configurations(n_spin_orb, nelec)



# ----------------- RBM -----------------
# Inicializa a RBM com α = 1 (n_h = n_v)
rbm = RBM(n_visible=12, n_hidden=12)
np.random.seed(42)
rbm.a = np.random.normal(0, 0.01, size=12)
rbm.b = np.random.normal(0, 0.01, size=12)
rbm.W = np.random.normal(0, 0.01, size=(12, 12))



# Treina a RBM
energias, a_hist, b_hist, W_hist = train_rbm_variacional(rbm, configs, theta, epochs=20, lr=0.001)



# Salva curva de energia
np.savetxt("convergencia_rbm_h2.txt", energias)


from pyscf import gto, scf

mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # distância em angstrom
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

E_nuclear = mol.energy_nuc()
print(f"🔬 Energia nuclear (núcleo-núcleo): {E_nuclear:.6f} Ha")


import matplotlib.pyplot as plt
import numpy as np

# Plota
def plot_energia(energias):
    plt.plot(energias+E_nuclear, label="\u2328 \u2328  <H> durante treino")
    plt.axhline(-1.728379+E_nuclear, color='gray', linestyle='--', label="Energia FCI")
    plt.title("Convergência da Energia com RBM para H₂ (reprodução do artigo)")
    plt.xlabel("Época")
    plt.ylabel("Energia (Ha)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergencia_h2_rbm_reproducao.png")
    plt.show()

def plot_parametros(a_hist, b_hist, W_hist):
    a_hist = np.array(a_hist)
    b_hist = np.array(b_hist)
    W_hist = np.array(W_hist)  # shape: (epochs, n_v, n_h)

    plt.figure(figsize=(14, 6))

    # Bias visíveis
    plt.subplot(1, 3, 1)
    for i in range(a_hist.shape[1]):
        plt.plot(a_hist[:, i], label=f"$a_{i}$")
    plt.title("Evolução dos Bias Visíveis (a)")
    plt.xlabel("Época")
    plt.grid(True)

    # Bias ocultos
    plt.subplot(1, 3, 2)
    for i in range(b_hist.shape[1]):
        plt.plot(b_hist[:, i], label=f"$b_{i}$")
    plt.title("Evolução dos Bias Ocultos (b)")
    plt.xlabel("Época")
    plt.grid(True)

    # Pesos (só alguns)
    plt.subplot(1, 3, 3)
    n_v, n_h = W_hist.shape[1], W_hist.shape[2]
    for i in range(n_v):
        for j in range(n_h):
            if i * n_h + j < 6:  # até 6 curvas no gráfico
                plt.plot(W_hist[:, i, j], label=f"$W_{{{i},{j}}}$")
    plt.title("Evolução dos Pesos (W)")
    plt.xlabel("Época")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_energia_avancada(energia_por_epoca, energia_fci=-8.509289, energia_nuc=0.734):
    energia_total = [e + energia_nuc for e in energia_por_epoca]
    energia_fci_total = energia_fci + energia_nuc
    epoca_convergencia = 8
    energia_final = energia_total[-1]

    plt.figure(figsize=(10, 6))
    
    # Curva da energia durante o treino
    plt.plot(energia_total, label=f"🧠 ⟨H⟩ durante treino\n(Eₙₑᵣgᵢₐ ₐₜᵤₐₗ = {energia_final:.6f} Ha)", color='steelblue')

    # Linha da energia FCI eletrônica + nuclear
    plt.axhline(energia_fci_total, linestyle='--', color='gray', label=f"Energia FCI + nuclear ({energia_fci_total:.6f} Ha)")

    # Linha vertical da convergência
    plt.axvline(epoca_convergencia, linestyle=':', color='black', label=f"Convergência (época {epoca_convergencia})")

    # Anotação da energia final
    plt.annotate(f"{energia_final:.6f} Ha",
                 xy=(epoca_convergencia, energia_final),
                 xytext=(epoca_convergencia - 3, energia_final + 0.03),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=10)

    plt.title("Convergência da Energia com RBM para H$_2$ (reprodução do artigo)")
    plt.xlabel("Época")
    plt.ylabel("Energia Total (Ha)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

E_nuclear=0
plot_energia_avancada(energias, energia_nuc=E_nuclear)
plot_parametros(a_hist, b_hist, W_hist)

