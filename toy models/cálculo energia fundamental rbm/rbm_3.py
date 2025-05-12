import numpy as np

import numpy as np
from itertools import combinations

# ----------------- RBM -----------------
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.random.randn(n_visible) * 0.01
        self.b = np.random.randn(n_hidden) * 0.01
        self.W = np.random.randn(n_visible, n_hidden) * 0.01

    def psi(self, sigma):
        preactivation = self.b + np.dot(sigma, self.W)
        preactivation = np.clip(preactivation, -10, 10)
        hidden_product = np.prod(2 * np.cosh(preactivation))
        a_dot = np.dot(self.a, sigma)
        a_dot = np.clip(a_dot, -20, 20)
        return np.exp(a_dot) * hidden_product

    def log_psi(self, sigma):
        a_dot = np.dot(self.a, sigma)
        preactivation = self.b + np.dot(sigma, self.W)
        cosh_terms = np.log(2 * np.cosh(preactivation))
        return a_dot + np.sum(cosh_terms)

    def log_derivatives(self, sigma):
        h_input = self.b + np.dot(sigma, self.W)
        tanh_h = np.tanh(h_input)
        return sigma, tanh_h, np.outer(sigma, tanh_h)

# ----------------- Configurações -----------------
def generate_configurations(n_sites, n_electrons):
    configs = []
    for occ in combinations(range(n_sites), n_electrons):
        config = np.zeros(n_sites)
        config[list(occ)] = 1
        configs.append(config)
    return np.array(configs)
    
def train_rbm_sr(rbm, configs, H_jw, epochs=300, lr=0.01, clip_value=2.0, tol=1e-3):
    energia_por_epoca = []

    n_param = rbm.n_visible + rbm.n_hidden + rbm.n_visible * rbm.n_hidden
    theta_vec = np.concatenate([rbm.a, rbm.b, rbm.W.flatten()])

    for epoch in range(epochs):
        psi_vals = np.array([rbm.psi(cfg) for cfg in configs])
        norm = np.sum(np.abs(psi_vals) ** 2)

        E_total = 0.0
        E_locals = []
        if () in H_jw.terms:
            E_total += H_jw.terms[()] * norm

        for i, sigma in enumerate(configs):
            psi_sigma = psi_vals[i]
            E_loc = 0
            for term, coeff in H_jw.terms.items():
                if term == (): continue
                new_sigma = sigma.copy()
                phase = 1.0 + 0j
                for qubit, op in term:
                    if op == 'Z':
                        phase *= (-1) ** int(new_sigma[qubit])
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

        Oks = []
        for sigma in configs:
            da, db, dW = rbm.log_derivatives(sigma)
            Oks.append(np.concatenate([da, db, dW.flatten()]))

        Oks = np.array(Oks)
        O_mean = np.sum((np.abs(psi_vals) ** 2)[:, None] * Oks, axis=0) / norm
        E_mean = np.real(E_mean)
        E_locals = np.array(E_locals)

        grads = np.zeros_like(O_mean)
        S_matrix = np.zeros((n_param, n_param))
        for i in range(len(configs)):
            O_i = Oks[i]
            diff_O = O_i - O_mean
            S_matrix += np.outer(diff_O, diff_O) * np.abs(psi_vals[i]) ** 2
            grads += (E_locals[i] - E_mean) * diff_O * np.abs(psi_vals[i]) ** 2

        S_matrix /= norm
        grads /= norm

        try:
            delta_theta = np.linalg.solve(S_matrix + 1e-4 * np.eye(n_param), grads)
        except np.linalg.LinAlgError:
            print("⚠️ Matriz S singular — usando gradiente direto")
            delta_theta = grads

        # Atualiza parâmetros
        theta_vec -= lr * delta_theta

        # Clipping para evitar explosão
        theta_vec = np.clip(theta_vec, -clip_value, clip_value)

        # Atribui os novos valores
        rbm.a = theta_vec[:rbm.n_visible]
        rbm.b = theta_vec[rbm.n_visible:rbm.n_visible + rbm.n_hidden]
        rbm.W = theta_vec[rbm.n_visible + rbm.n_hidden:].reshape(rbm.n_visible, rbm.n_hidden)

        print(f"Epoch {epoch+1:03d} → ⟨H⟩ = {E_mean:.6f} Ha")

        # Critério de convergência por precisão química
        if epoch > 5:
            delta_E = np.abs(energia_por_epoca[-1] - energia_por_epoca[-2])
            if delta_E < tol:
                print(f"✅ Convergência atingida com ΔE = {delta_E:.2e} Ha na época {epoch+1}")
                break

    return energia_por_epoca