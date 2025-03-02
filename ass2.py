import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0.05, 0.10]])

Y = np.array([[0.01, 0.99]])


W_hidden = np.array([[0.15, 0.25], [0.20, 0.30]])


W_output = np.array([[0.40, 0.50], [0.45, 0.55]])


b_hidden = np.array([0.35, 0.35])
b_output = np.array([0.60, 0.60])


learning_rate = 0.5


epochs = 10000

for epoch in range(epochs):
    
    net_hidden = np.dot(X, W_hidden) + b_hidden
    h = sigmoid(net_hidden)
    
    net_output = np.dot(h, W_output) + b_output
    o = sigmoid(net_output)
    

    error = 0.5 * np.sum((Y - o) ** 2)
    

    delta_output = (o - Y) * sigmoid_derivative(o)
    delta_hidden = np.dot(delta_output, W_output.T) * sigmoid_derivative(h)
    
    W_output -= learning_rate * np.dot(h.T, delta_output)
    W_hidden -= learning_rate * np.dot(X.T, delta_hidden)
    
    b_output -= learning_rate * delta_output.sum(axis=0)
    b_hidden -= learning_rate * delta_hidden.sum(axis=0)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")


print("\nFinal Output after training:")
print(o)
