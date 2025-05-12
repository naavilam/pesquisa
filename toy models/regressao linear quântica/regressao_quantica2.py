import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Dados sintéticos: função curva!
X = np.linspace(0, np.pi, 30)
X_norm = qml.numpy.array(X / np.pi)
Y = qml.numpy.array(np.sin(2 * X) + X)

# Dispositivo
dev = qml.device("default.qubit", wires=1)

# Novo circuito com mais expressividade
@qml.qnode(dev)
def circuit(x, weights):
    qml.RX(x * np.pi, wires=0)      # codificação da entrada
    qml.RY(weights[0], wires=0)     # camada 1
    qml.RZ(weights[1], wires=0)
    qml.RY(weights[2], wires=0)     # camada 2
    return qml.expval(qml.PauliZ(0))

# Modelo com rescale
scale = -3.0
shift = 2.5
def model(x, weights):
    return scale * circuit(x, weights) + shift

# Custo
def cost(weights):
    preds = qml.numpy.array([model(x, weights) for x in X_norm])
    return qml.numpy.mean((preds - Y)**2)

# Inicialização
weights = qml.numpy.array([0.01, 0.01, 0.01], requires_grad=True)
opt = qml.optimize.AdamOptimizer(stepsize=0.05)

# Treinamento
losses = []
for i in range(150):
    weights = opt.step(cost, weights)
    l = cost(weights)
    losses.append(l)
    if i % 20 == 0:
        print(f"Interation {i:3d} | Loss: {l:.6f}")

# Visualização
preds = [model(x, weights) for x in X_norm]
plt.figure(figsize=(8, 4))
plt.plot(X, Y, "o", label="Target function: sin(2x) + x")
plt.plot(X, preds, "-", label="Fitted quantum model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Curve Fitting with a Variational Quantum Circuit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()