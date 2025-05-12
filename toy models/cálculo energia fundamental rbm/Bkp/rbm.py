import numpy as np

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



class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.random.randn(n_visible) * 0.01  # bias visível
        self.b = np.random.randn(n_hidden) * 0.01   # bias oculto
        self.W = np.random.randn(n_visible, n_hidden) * 0.01  # pesos

    def psi(self, sigma):
        """
        Retorna a amplitude (não normalizada) da função de onda Ψ(sigma)
        """
        preactivation = self.b + np.dot(sigma, self.W)
        hidden_product = np.prod(2 * np.cosh(preactivation))
        return np.exp(np.dot(self.a, sigma)) * hidden_product



# Sistema com 4 orbitais → 4 qubits (spin orbitais)
rbm = RBM(n_visible=4, n_hidden=4)

# Todas as configurações possíveis com dois elétrons
from itertools import combinations
def generate_configurations(n_sites, n_electrons):
    configs = []
    for occ in combinations(range(n_sites), n_electrons):
        config = np.zeros(n_sites)
        config[list(occ)] = 1
        configs.append(config)
    return np.array(configs)

configs = generate_configurations(4, 2)

# Hamiltoniano simplificado: energia orbital (só para exemplo)
orbital_energies = np.array([1.0, 2.0, 3.0, 4.0])
def energy(config):
    return np.sum(orbital_energies * config)

# Esperança de energia com RBM (não normalizado)
numerador = 0
denominador = 0
for cfg in configs:
    psi_val = rbm.psi(cfg)
    e_val = energy(cfg)
    numerador += np.abs(psi_val)**2 * e_val
    denominador += np.abs(psi_val)**2

E_rbm = numerador / denominador
print(f"⟨H⟩ com RBM: {E_rbm:.6f} Ha")


def log_derivatives(rbm, sigma):
    # Derivadas do log(Ψ) em relação aos parâmetros
    h_input = rbm.b + np.dot(sigma, rbm.W)
    tanh_h = np.tanh(h_input)

    d_a = sigma
    d_b = tanh_h
    d_W = np.outer(sigma, tanh_h)

    return d_a, d_b, d_W

def train_rbm(rbm, configs, energy_fn, lr=0.05, epochs=100):
    energia_por_epoca = []
    for epoch in range(epochs):
        psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
        E_locals = np.array([energy_fn(cfg) for cfg in configs])
        probs = np.abs(psi_vals)**2
        Z = np.sum(probs)
        norm_probs = probs / Z
        E_mean = np.sum(norm_probs * E_locals)

        # Gradientes acumulados
        grad_a = np.zeros_like(rbm.a)
        grad_b = np.zeros_like(rbm.b)
        grad_W = np.zeros_like(rbm.W)

        for i, sigma in enumerate(configs):
            d_a, d_b, d_W = log_derivatives(rbm, sigma)
            weight = norm_probs[i]
            E_loc = E_locals[i]
            grad_a += weight * (E_loc - E_mean) * d_a
            grad_b += weight * (E_loc - E_mean) * d_b
            grad_W += weight * (E_loc - E_mean) * d_W

        # Atualização dos parâmetros
        rbm.a -= lr * grad_a
        rbm.b -= lr * grad_b
        rbm.W -= lr * grad_W

        energia_por_epoca.append(E_mean)
        print(f"Epoch {epoch+1:03d} → ⟨H⟩: {E_mean:.6f} Ha")
    return energia_por_epoca





def apply_pauli_term(term, config):
    """
    Aplica um termo de Pauli (como ('X0', 'Y1')) à configuração clássica e retorna:
    - nova configuração (se não for zero)
    - fase complexa multiplicativa
    """
    new_config = config.copy()
    coef = 1.0 + 0j

    for op in term:
        if op[0] == 'I':
            continue
        qubit = int(op[1:])
        if op[0] == 'Z':
            coef *= (-1)**int(config[qubit])
        elif op[0] == 'X' or op[0] == 'Y':
            # Flipa o bit
            new_config[qubit] = 1 - new_config[qubit]
            if op[0] == 'Y':
                coef *= (1j if config[qubit] == 0 else -1j)

    return new_config, coef


import numpy as np

def expectation_rbm_openfermion(H, rbm, configs):
    psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
    norm = np.sum(np.abs(psi_vals) ** 2)
    E_total = 0

    # Primeiro soma o termo constante (se houver)
    if () in H.terms:
        E_total += H.terms[()] * norm  # só uma vez!

    # Agora percorre as configs e os outros termos
    for i, sigma in enumerate(configs):
        psi_sigma = psi_vals[i]
        for term, coeff in H.terms.items():
            if term == ():  # já tratamos antes
                continue
            new_sigma = sigma.copy()
            phase = 1.0 + 0j

            for qubit, op in term:
                if op == 'Z':
                    phase *= (-1) ** int(new_sigma[qubit])
                elif op == 'X' or op == 'Y':
                    new_sigma[qubit] = 1 - new_sigma[qubit]
                    if op == 'Y':
                        phase *= (1j if sigma[qubit] == 0 else -1j)

            try:
                j = configs.tolist().index(new_sigma.tolist())
                psi_sp = psi_vals[j]
                E_total += coeff * phase * np.conj(psi_sigma) * psi_sp
            except ValueError:
                continue

    return (E_total / norm).real



def energia_local_rbm_factory(H_jw, rbm, configs):
    psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
    def energia_local(cfg):
        idx = configs.tolist().index(cfg.tolist())
        psi_sigma = psi_vals[idx]
        E = 0
        for term, coeff in H_jw.terms.items():
            new_cfg = cfg.copy()
            phase = 1.0 + 0j
            for qubit, op in term:
                if op == 'Z':
                    phase *= (-1) ** int(new_cfg[qubit])
                elif op == 'X' or op == 'Y':
                    new_cfg[qubit] = 1 - new_cfg[qubit]
                    if op == 'Y':
                        phase *= (1j if cfg[qubit] == 0 else -1j)
            try:
                j = configs.tolist().index(new_cfg.tolist())
                psi_sp = psi_vals[j]
                if term == ():
                    E += coeff * np.abs(psi_sigma) ** 2
                else:
                    E += coeff * phase * np.conj(psi_sigma) * psi_sp
            except ValueError:
                continue
        return E.real
    return energia_local


def train_rbm_openfermion(rbm, configs, H, lr=0.05, epochs=100):
    energia_por_epoca = []

    for epoch in range(epochs):
        psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
        norm = np.sum(np.abs(psi_vals) ** 2)

        # Calcula energia total com Hamiltoniano real
        E_total = 0
        if () in H.terms:
            E_total += H.terms[()] * norm

        E_locals = []

        for i, sigma in enumerate(configs):
            psi_sigma = psi_vals[i]
            E_loc = 0

            for term, coeff in H.terms.items():
                if term == (): continue
                new_sigma = sigma.copy()
                phase = 1.0 + 0j
                for qubit, op in term:
                    if op == 'Z':
                        phase *= (-1) ** int(new_sigma[qubit])
                    elif op == 'X' or op == 'Y':
                        new_sigma[qubit] = 1 - new_sigma[qubit]
                        if op == 'Y':
                            phase *= (1j if sigma[qubit] == 0 else -1j)
                try:
                    j = configs.tolist().index(new_sigma.tolist())
                    psi_sp = psi_vals[j]
                    contrib = coeff * phase * np.conj(psi_sigma) * psi_sp
                    E_total += contrib
                    E_loc += contrib
                except ValueError:
                    continue

            E_locals.append(E_loc.real)

        E_mean = (E_total / norm).real
        energia_por_epoca.append(E_mean)

        # Gradientes
        grad_a = np.zeros_like(rbm.a)
        grad_b = np.zeros_like(rbm.b)
        grad_W = np.zeros_like(rbm.W)
        probs = np.abs(psi_vals)**2
        norm_probs = probs / norm

        for i, sigma in enumerate(configs):
            d_a, d_b, d_W = log_derivatives(rbm, sigma)
            weight = norm_probs[i]
            E_loc = E_locals[i]
            grad_a += weight * (E_loc - E_mean) * d_a
            grad_b += weight * (E_loc - E_mean) * d_b
            grad_W += weight * (E_loc - E_mean) * d_W

        rbm.a -= lr * grad_a
        rbm.b -= lr * grad_b
        rbm.W -= lr * grad_W

        print(f"Epoch {epoch+1:03d} → ⟨H⟩: {E_mean:.6f} Ha")

    return energia_por_epoca
    

n_qubits=4
n_electrons=2

# Geração de todas as configs com 2 elétrons em 4 orbitais
configs = generate_configurations(n_qubits, n_electrons)  # ex: 4, 2
E_rbm_real = expectation_rbm_openfermion(H_jw, rbm, configs)
print(f"⟨H₂⟩ com RBM e JW: {E_rbm_real:.6f} Ha")




energy_fn = energia_local_rbm_factory(H_jw, rbm, configs)



energias = train_rbm_openfermion(rbm, configs, H_jw, lr=0.05, epochs=100)

# Energia esperada
# E_rbm = expectation_rbm(H_jw, rbm, configs)
# print(f"⟨H⟩ (Hamiltoniano real de H₂): {E_rbm:.6f} Ha")







import matplotlib.pyplot as plt

# Supondo que você tenha uma lista chamada `energia_por_epoca`
# que você foi preenchendo durante o treinamento da RBM:
# Exemplo:
# energia_por_epoca = []
# ...
# for epoch in range(epochs):
#     ...
#     energia_por_epoca.append(E_mean)

def plot_energia(energia_por_epoca):
    plt.figure(figsize=(10, 5))
    plt.plot(energia_por_epoca, label='⟨H⟩ durante treino', color='mediumslateblue')
    plt.axhline(energia_por_epoca[-1], linestyle='--', color='gray', label='Energia final')
    plt.xlabel('Época')
    plt.ylabel('Energia ⟨H⟩ (Ha)')
    plt.title('Convergência da Energia da RBM')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Exemplo de uso no final do seu script:
plot_energia(energias)