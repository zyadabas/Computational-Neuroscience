import numpy as np

def tanh(x):
    return np.tanh(x)

def forward_pass(i1, i2, weights, biases):
    net_h1 = i1 * weights['w1'] + i2 * weights['w2'] + biases['b1']
    net_h2 = i1 * weights['w3'] + i2 * weights['w4'] + biases['b2']
    
    out_h1 = tanh(net_h1)
    out_h2 = tanh(net_h2)
    
    net_o1 = out_h1 * weights['w5'] + out_h2 * weights['w6']
    net_o2 = out_h1 * weights['w7'] + out_h2 * weights['w8']
    
    out_o1 = tanh(net_o1)
    out_o2 = tanh(net_o2)
    
    return out_o1, out_o2

np.random.seed(42)
weights = {
    'w1': np.random.uniform(-0.5, 0.5),
    'w2': np.random.uniform(-0.5, 0.5),
    'w3': np.random.uniform(-0.5, 0.5),
    'w4': np.random.uniform(-0.5, 0.5),
    'w5': np.random.uniform(-0.5, 0.5),
    'w6': np.random.uniform(-0.5, 0.5),
    'w7': np.random.uniform(-0.5, 0.5),
    'w8': np.random.uniform(-0.5, 0.5)
}

biases = {'b1': 0.5, 'b2': 0.7}

i1, i2 = 0.6, -0.4  

output = forward_pass(i1, i2, weights, biases)
print("Network Output:", output)
