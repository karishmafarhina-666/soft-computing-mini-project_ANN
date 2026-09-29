import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Dataset
X_train = [
    [1.0, 2.0], [2.0, 1.5], [1.5, 3.0], [3.0, 2.0], [1.0, 1.0], [2.5, 2.5], 
    [8.0, 9.0], [9.0, 7.5], [7.5, 8.0], [9.0, 9.0], [8.5, 8.5], [8.0, 7.0], 
    [6.0, 7.0], [7.0, 6.0], [3.0, 8.0], [8.0, 3.0], [5.0, 5.0], [4.5, 6.5]  
]
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

# Train the Perceptron
def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

random.seed(42)
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)
learning_rate = 0.1

for _ in range(2000):
    for i in range(len(X_train)):
        x1, x2 = X_train[i]
        y_true = y_train[i]
        y_pred = sigmoid((w1 * x1) + (w2 * x2) + b)
        error = y_pred - y_true
        w1 -= learning_rate * error * x1
        w2 -= learning_rate * error * x2
        b -= learning_rate * error

# Create Visualization
print("Generating visualization...")
X_np = np.array(X_train)
y_np = np.array(y_train)

# Create a mesh grid for the background colors
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Calculate probabilities for the entire grid
Z = (w1 * xx) + (w2 * yy) + b
Z_prob = 1 / (1 + np.exp(-Z))
Z_pred = Z_prob >= 0.5

# Plotting
plt.figure(figsize=(8, 6))

# Plot the background decision space
plt.contourf(xx, yy, Z_prob, levels=50, cmap="RdYlGn_r", alpha=0.3)
plt.colorbar(label='Probability of Fault (1.0 = High, 0.0 = Low)')

# Plot the exact decision boundary (where w1*x1 + w2*x2 + b = 0)
x_boundary = np.array([x_min, x_max])
y_boundary = -(w1 * x_boundary + b) / w2
plt.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary (0.5)')

# Plot the actual data points
healthy = X_np[y_np == 0]
faulty = X_np[y_np == 1]

plt.scatter(healthy[:, 0], healthy[:, 1], c='green', marker='o', edgecolors='k', s=100, label='Healthy (0)')
plt.scatter(faulty[:, 0], faulty[:, 1], c='orange', marker='^', edgecolors='k', s=100, label='Faulty (1)')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Single-Layer Perceptron: IoT Machine Fault Classifier')
plt.xlabel('Temperature Severity (0-10)')
plt.ylabel('Vibration Intensity (0-10)')
plt.legend(loc='lower right')

# Save and show
plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
print("Saved to decision_boundary.png")
plt.show()
