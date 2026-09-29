import math
import random

# 1. The Dataset [Temperature Severity, Vibration Intensity]
X_train = [
    [1.0, 2.0], [2.0, 1.5], [1.5, 3.0], [3.0, 2.0], [1.0, 1.0], [2.5, 2.5], # Healthy
    [8.0, 9.0], [9.0, 7.5], [7.5, 8.0], [9.0, 9.0], [8.5, 8.5], [8.0, 7.0], # Faulty
    [6.0, 7.0], [7.0, 6.0], [3.0, 8.0], [8.0, 3.0], [5.0, 5.0], [4.5, 6.5]  # Boundary
]

# Labels: 0 = Healthy, 1 = Faulty
y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

# 2. Activation Function
def sigmoid(z):
    # Clamping z to avoid math overflow errors
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

# 3. Model Initialization
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

learning_rate = 0.1
epochs = 2000

print("Starting training...")

# 4. Training Loop (Backpropagation / Gradient Descent)
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(X_train)):
        x1, x2 = X_train[i]
        y_true = y_train[i]
        
        # Forward pass: weighted sum
        z = (w1 * x1) + (w2 * x2) + b
        
        # Output probability
        y_pred = sigmoid(z)
        
        # Calculate loss (Binary Cross-Entropy)
        # Adding a tiny epsilon to avoid log(0)
        epsilon = 1e-15
        loss = - (y_true * math.log(y_pred + epsilon) + (1 - y_true) * math.log(1 - y_pred + epsilon))
        total_loss += loss
        
        # Gradient calculation (derivative of BCE loss + sigmoid)
        error = y_pred - y_true
        
        # Weight update
        w1 -= learning_rate * error * x1
        w2 -= learning_rate * error * x2
        b -= learning_rate * error
        
    if epoch % 400 == 0 or epoch == epochs - 1:
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch:4d} | Error: {avg_loss:.4f}")

print("\nTraining Complete!")
print(f"Final Synaptic Weights: w1={w1:.4f}, w2={w2:.4f}, bias={b:.4f}\n")

# 5. Testing on New Machine Data
print("Testing on unseen machine readings:")
test_cases = [
    ([2.0, 2.0], "Expected Healthy"),
    ([9.0, 8.5], "Expected Faulty"),
    ([5.5, 6.0], "Unknown/Boundary")
]

for features, expectation in test_cases:
    x1, x2 = features
    z = (w1 * x1) + (w2 * x2) + b
    prob = sigmoid(z)
    prediction = 1 if prob >= 0.5 else 0
    status = "Faulty" if prediction == 1 else "Healthy"
    
    print(f"Temp: {x1}, Vib: {x2} -> Prob: {prob:.4f} => Class: {status} ({expectation})")
