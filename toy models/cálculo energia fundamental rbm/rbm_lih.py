import numpy as np
import matplotlib.pyplot as plt
from openfermion import FermionOperator, jordan_wigner
from itertools import combinations

# ----------------- Classe RBM -----------------
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.random.randn(n_visible) * 0.01
        self.b = np.random.randn(n_hidden) * 0.01
        self.W = np.random.randn(n_visible, n_hidden) * 0.01

    # def psi(self, sigma):
    #     preactivation = self.b + np.dot(sigma, self.W)
    #     preactivation = np.clip(preactivation, -30, 30)  # evita overflow
    #     hidden_product = np.prod(2 * np.cosh(preactivation))
    #     a_dot = np.dot(self.a, sigma)
    #     a_dot = np.clip(a_dot, -50, 50)
    #     return np.exp(a_dot) * hidden_product

    def log_psi(self, sigma):
        """Retorna o logaritmo da função de onda log(Ψ(σ))"""
        a_dot = np.dot(self.a, sigma)
        preactivation = self.b + np.dot(sigma, self.W)
        cosh_terms = np.log(2 * np.cosh(preactivation))
        return a_dot + np.sum(cosh_terms)


    def psi(self, sigma):
        """Retorna Ψ(σ) a partir de log(Ψ(σ))"""
        return np.exp(self.log_psi(sigma))
        # preactivation = self.b + np.dot(sigma, self.W)
        # # preactivation = np.clip(preactivation, -10, 10)
        # hidden_product = np.prod(2 * np.cosh(preactivation))
        # a_dot = np.dot(self.a, sigma)
        # # a_dot = np.clip(a_dot, -20, 20)
        # return np.exp(a_dot) * hidden_product

    def log_derivatives(self, sigma):
        h_input = self.b + np.dot(sigma, self.W)
        tanh_h = np.tanh(h_input)
        return sigma, tanh_h, np.outer(sigma, tanh_h)

# ----------------- Geração de configurações -----------------
def generate_configurations(n_sites, n_electrons):
    configs = []
    for occ in combinations(range(n_sites), n_electrons):
        config = np.zeros(n_sites)
        config[list(occ)] = 1
        configs.append(config)
    return np.array(configs)

# ----------------- Treinamento variacional -----------------
def train_rbm_variacional(rbm, configs, H_jw, epochs=100, lr=0.05):
    energia_por_epoca = []
    # Novo: armazenar a jornada dos parâmetros
    a_hist = []
    b_hist = []
    W_hist = []
    initial_lr = lr
    gamma = 0.9         # fator de decaimento
    decay_interval = 2 # a cada 10 épocas

    for epoch in range(epochs):
        # Dentro do loop for epoch in range(epochs):
        lr = initial_lr * (gamma ** (epoch // decay_interval))
        psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
        norm = np.sum(np.abs(psi_vals)**2)

        E_total = 0
        if () in H_jw.terms:
            E_total += H_jw.terms[()] * norm

        E_locals = []
        for i, sigma in enumerate(configs):
            psi_sigma = psi_vals[i]
            E_loc = 0
            for term, coeff in H_jw.terms.items():
                if term == (): continue
                new_sigma = sigma.copy()
                phase = 1.0 + 0j
                for qubit, op in term:
                    if op == 'Z':
                        phase *= (-1)**int(new_sigma[qubit])
                    elif op in ('X', 'Y'):
                        new_sigma[qubit] = 1 - new_sigma[qubit]
                        if op == 'Y':
                            phase *= 1j if sigma[qubit] == 0 else -1j
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

        if epoch == 0:
            print(f"[Check] Energia inicial (sem treino): {E_mean:.6f} Ha")


        # zera o valor dos gradientes
        grad_a = np.zeros_like(rbm.a)
        grad_b = np.zeros_like(rbm.b)
        grad_W = np.zeros_like(rbm.W)

        #         # Clipping de gradientes
        # max_grad = 5.0
        # grad_a = np.clip(grad_a, -max_grad, max_grad)
        # grad_b = np.clip(grad_b, -max_grad, max_grad)
        # grad_W = np.clip(grad_W, -max_grad, max_grad)


        # calcula a probabilidade de P(psi) = |psi|ˆ2
        probs = np.abs(psi_vals)**2
        norm_probs = probs / norm

        for i, sigma in enumerate(configs):
            d_a, d_b, d_W = rbm.log_derivatives(sigma)
            weight = norm_probs[i]
            E_loc = E_locals[i]
            grad_a += weight * (E_loc - E_mean) * d_a
            grad_b += weight * (E_loc - E_mean) * d_b
            grad_W += weight * (E_loc - E_mean) * d_W

        rbm.a -= lr * grad_a
        rbm.b -= lr * grad_b
        rbm.W -= lr * grad_W

        if epoch % 1 == 0:  # muda o intervalo de rastreio se quiser
            delta_a = lr * grad_a
            delta_b = lr * grad_b
            delta_W = lr * grad_W

            print(f"\n🔍 >>>>>>>>> Epoch {epoch+1}")
            print("🔹 W max:", np.max(np.abs(rbm.W)))
            print("🔹 psi[0]:", rbm.psi(configs[0]))

            print("\n--- Bias Visíveis (a) ---")
            print("a        =", rbm.a)
            print("grad_a   =", grad_a)
            print("Δa       =", delta_a)
            print("a (nova) =", rbm.a - delta_a)

            print("\n--- Bias Ocultos (b) ---")
            print("b        =", rbm.b)
            print("grad_b   =", grad_b)
            print("Δb       =", delta_b)
            print("b (nova) =", rbm.b - delta_b)

            print("\n--- Pesos (W) ---")
            print("W        =\n", rbm.W)
            print("grad_W   =\n", grad_W)
            print("ΔW       =\n", delta_W)
            print("W (nova) =\n", rbm.W - delta_W)

            print(f"\n⚡ Energia média ⟨H⟩: {E_mean:.6f} Ha")
            print("-" * 50)

        max_norm = 2.0  # limite para os pesos
        rbm.W = np.clip(rbm.W, -max_norm, max_norm)
        rbm.a = np.clip(rbm.a, -max_norm, max_norm)
        rbm.b = np.clip(rbm.b, -max_norm, max_norm)
        a_hist.append(rbm.a.copy())
        b_hist.append(rbm.b.copy())
        W_hist.append(rbm.W.copy())
        print(f"Epoch {epoch+1:03d} → ⟨H⟩: {E_mean:.6f} Ha")
 

    return energia_por_epoca, a_hist, b_hist, W_hist

# ----------------- Gráfico -----------------
def plot_energia(energia_por_epoca):
    plt.figure(figsize=(10, 5))
    plt.plot(energia_por_epoca, label="⟨H⟩ durante o treino", color='mediumblue')
    plt.axhline(energia_por_epoca[-1], linestyle='--', color='gray', label='Energia final')
    plt.xlabel("Época")
    plt.ylabel("Energia ⟨H⟩ (Ha)")
    plt.title("Convergência da Energia com RBM Variacional")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()












