import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, np.pi, 30)
X_norm = qml.numpy.array(X / np.pi)
Y = qml.numpy.array(np.sin(2 * X) + X)

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x, weights):
    qml.RX(x * np.pi, wires=0)
    qml.RY(weights[0], wires=0)
    qml.RZ(weights[1], wires=0)
    qml.RY(weights[2], wires=0)
    qml.RZ(weights[3], wires=0)  # segunda "camada"
    return qml.expval(qml.PauliZ(0))

scale = -3.0
shift = 2.5

def model(x, weights):
    return scale * circuit(x, weights) + shift

def cost(weights):
    preds = qml.numpy.array([model(x, weights) for x in X_norm])
    return qml.numpy.mean((preds - Y)**2)

weights = qml.numpy.array([0.01, 0.01, 0.01, 0.01], requires_grad=True)
opt = qml.optimize.AdamOptimizer(stepsize=0.05)

losses = []
for i in range(250):
    weights = opt.step(cost, weights)
    l = cost(weights)
    losses.append(l)
    if i % 25 == 0:
        print(f"Iteração {i:3d} | Loss: {l:.6f}")

preds = [model(x, weights) for x in X_norm]
plt.figure(figsize=(8, 4))
plt.plot(X, Y, 'o', label="Função alvo: $\sin(2x) + x$")
plt.plot(X, preds, '-', label="Predição VQC (2 camadas)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Aumentando a Expressividade com Mais Parâmetros")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()