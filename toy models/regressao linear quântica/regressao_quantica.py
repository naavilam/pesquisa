import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

# Dados sintéticos
X = np.linspace(0, np.pi, 20)
X_norm = qml.numpy.array(X / np.pi)                  # entrada normalizada
Y = qml.numpy.array(2 * X + 1)                       # saída esperada

# Dispositivo com 1 qubit
dev = qml.device("default.qubit", wires=1)

# Circuito quântico variacional
@qml.qnode(dev)
def circuit(x, weights):
    qml.RX(x * np.pi, wires=0)      # codificação da entrada
    qml.RY(weights[0], wires=0)     # parâmetro treinável
    return qml.expval(qml.PauliZ(0))  # ⟨Z⟩ ∈ [-1, 1]

# Modelo escalonado para saída linear
scale = -4.0
shift = 5.0

def model(x, weights):
    return scale * circuit(x, weights) + shift

# Função custo
def cost(weights):
    predictions = qml.numpy.array([model(x, weights) for x in X_norm])
    return qml.numpy.mean((predictions - Y) ** 2)

# Inicialização do parâmetro
weights = qml.numpy.array([0.1], requires_grad=True)

# Otimizador
opt = qml.optimize.NesterovMomentumOptimizer(stepsize=0.1)
losses = []

# Treinamento
for i in range(100):
    weights = opt.step(cost, weights)
    l = cost(weights)
    losses.append(l)
    if i % 10 == 0:
        print(f"Iteraction {i:3d} | Loss: {l:.6f} | Weight: {weights[0]:.4f}")

# Predição final
preds = [model(x, weights) for x in X_norm]

# Plot do resultado
plt.figure(figsize=(8, 4))
plt.plot(X, Y, "o", label="Target data")
plt.plot(X, preds, "-", label="Learnded Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Using a Variational Quantum Circuit")
plt.grid(False)
plt.tight_layout()
plt.show()