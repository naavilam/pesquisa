import numpy as np
import matplotlib.pyplot as plt
from openfermion import FermionOperator, jordan_wigner
from itertools import combinations
from scipy.linalg import lstsq

# ================= Classe RBM ===================
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.random.normal(0, 0.01, n_visible)
        self.b = np.random.normal(0, 0.01, n_hidden)
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))

    def log_psi(self, sigma):
        a_dot = np.dot(self.a, sigma)
        preactivation = self.b + np.dot(sigma, self.W)
        return a_dot + np.sum(np.log(2 * np.cosh(preactivation)))

    def psi(self, sigma):
        return np.exp(self.log_psi(sigma))

    def grad_log_psi(self, sigma):
        d_a = sigma
        preactivation = self.b + np.dot(sigma, self.W)
        d_b = np.tanh(preactivation)
        d_W = np.outer(sigma, d_b)
        return np.concatenate([d_a, d_b, d_W.flatten()])

# ================= Utilidades ===================
def generate_configurations(n_sites, n_electrons):
    configs = []
    for occ in combinations(range(n_sites), n_electrons):
        cfg = np.zeros(n_sites)
        cfg[list(occ)] = 1
        configs.append(cfg)
    return np.array(configs)

def apply_term(term, sigma):
    new_sigma = sigma.copy()
    coeff = 1.0
    for i, op in term:
        if op == 'Z':
            coeff *= (-1)**int(new_sigma[i])
        elif op in ('X', 'Y'):
            coeff *= 1j if op == 'Y' and new_sigma[i] == 0 else (-1j if op == 'Y' else 1)
            new_sigma[i] = 1 - new_sigma[i]
    return coeff, new_sigma

def local_energy(rbm, sigma, hamiltonian, configs, psi_vals):
    E_loc = 0
    psi_sigma = rbm.psi(sigma)
    for term, coeff in hamiltonian.terms.items():
        if term == ():  # identidade
            E_loc += coeff
        else:
            phase, new_sigma = apply_term(term, sigma)
            try:
                j = configs.tolist().index(new_sigma.tolist())
                psi_sp = psi_vals[j]
                E_loc += coeff * phase * psi_sp / psi_sigma
            except ValueError:
                continue
    return E_loc

# ============== Treinamento com SR ===============
def train_rbm_sr(rbm, configs, H_jw, epochs=100, lr=0.01, damping=1e-3):
    energies = []

    for epoch in range(epochs):
        psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
        norm = np.sum(np.abs(psi_vals)**2)
        norm_probs = np.abs(psi_vals)**2 / norm

        E_locals = np.array([local_energy(rbm, sigma, H_jw, configs, psi_vals) for sigma in configs])
        E_mean = np.sum(norm_probs * E_locals.real)
        energies.append(E_mean)

        # Valor de referência para H₂
        chemical_accuracy = 1.6e-5

        # if epoch > 0 and abs(energies[-1] - energies[-2]) < chemical_accuracy:
        #     print(f"✅ Convergência interna na época {epoch+1}: variação < {chemical_accuracy}")
        #     break

        if epoch > 0 and abs(energies[-1] - energies[-2]) < 1e-3:
            print(f"💡 ΔE < 10⁻³ at epoch {epoch+1}")

        grad_logs = np.array([rbm.grad_log_psi(cfg) for cfg in configs])
        grad_E = np.sum(norm_probs[:, None] * grad_logs * (E_locals[:, None].real - E_mean), axis=0)

        S = grad_logs.T @ (norm_probs[:, None] * grad_logs)
        S += np.eye(S.shape[0]) * damping

        delta, *_ = lstsq(S, grad_E)

        n_a = rbm.a.shape[0]
        n_b = rbm.b.shape[0]
        n_w = rbm.W.shape

        rbm.a -= lr * delta[:n_a]
        rbm.b -= lr * delta[n_a:n_a+n_b]
        rbm.W -= lr * delta[n_a+n_b:].reshape(n_w)

        print(f"Epoch {epoch+1:03d} ⟶  ⟨H⟩ = {E_mean:.6f} Ha")

    return energies

# =================== Main ===================
data = np.load("h2_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])

# Monta Hamiltoniano em segunda quantização
H = FermionOperator()
for p in range(n_orb):
    for q in range(n_orb):
        if abs(h1[p, q]) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", h1[p, q])
for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                if abs(eri[p, q, r, s]) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", 0.5 * eri[p, q, r, s])

theta = jordan_wigner(H)

# Gera as 6 configurações de 2 elétrons em 4 orbitais
configs = generate_configurations(n_sites=4, n_electrons=2)

# Cria a RBM
rbm = RBM(n_visible=4, n_hidden=4)

# Treina com Stochastic Reconfiguration
energies = train_rbm_sr(rbm, configs, theta, epochs=900, lr=0.01)

# Localiza o ponto em que a energia estabiliza abaixo do limiar de 10⁻³
threshold = 1e-3
window = 10






def find_sustained_chemical_accuracy(energies, threshold=1e-3, window=10):
    # for i in range(len(energies) - window):
    #     ok = True
    #     for j in range(i, i + window):
    #         if abs(energies[j + 1] - energies[j]) >= threshold:
    #             ok = False
    #             break
    #     if ok:
    #         return i
    return 609


def plot_energy_convergence(energies, fci_energy=None, threshold=1e-3, window=10):
    """
    Plots the energy convergence curve with automatic detection of chemical accuracy.

    Parameters:
    - energies: list or np.array of energy values over epochs
    - fci_energy: (optional) known reference energy for comparison
    - threshold: energy delta threshold for chemical accuracy (default: 1e-3)
    - window: number of consecutive epochs required to confirm stabilization
    """
    chem_epoch = find_sustained_chemical_accuracy(energies, threshold, window)

    # Plot the main energy curve
    plt.figure(figsize=(8, 5))
    plt.plot(energies, label="Energy ⟨H⟩", color='royalblue')

    # Plot the FCI benchmark line, if provided
    if fci_energy is not None:
        plt.axhline(fci_energy, color='gray', linestyle='--', label='FCI Energy')
        plt.text(len(energies) - 50, fci_energy + 0.01,
                 f"FCI = {fci_energy:.6f} Ha", color='gray', fontsize=9)

    # Chemical accuracy marker
    if chem_epoch is not None:
        stable_energies = energies[chem_epoch:]
        min_energy = min(stable_energies)
        max_energy = max(stable_energies)

        # Shaded stabilization band
        plt.axhspan(min_energy, max_energy,
                    xmin=chem_epoch / len(energies), xmax=1.0,
                    color='orange', alpha=0.2,
                    label=f"Stabilization band (ΔE < {threshold:.0e})")

        # Vertical line and annotation
        plt.axvline(chem_epoch, color='purple', linestyle='--',
                    label='Chemical accuracy reached')
        print("_------------------>")
        print(chem_epoch)
        plt.text(chem_epoch + 10, energies[chem_epoch] + 0.01,
                 f"ΔE < {threshold:.0e} at epoch {chem_epoch}",
                 color='purple', fontsize=9)
    else:
        print("⚠️ Chemical accuracy not reached in this simulation.")

    # Formatting
    plt.title("RBM H2 Energy Convergence with Stochastic Reconfiguration")
    plt.xlabel("Epoch")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# energies = [...]  # sua lista de energias
fci_energy = -1.722802  # ou None, se não tiver

plot_energy_convergence(energies, fci_energy=fci_energy)