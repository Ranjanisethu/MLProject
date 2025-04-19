import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # Flatten and normalize
y_train = y_train.reshape(-1, 1)

# OneHotEncode the labels
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train)

# Use only 1000 samples for faster training (can be more if your system supports)
X, y = x_train[:1000], y_train[:1000]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

# ReLU and its derivative (used instead of Sigmoid in hidden layers)
def relu(x):
    return np.maximum(0, x)

def derivatives_relu(x):
    return np.where(x > 0, 1, 0)

# Initialize network parameters
input_neurons = 784      # 28x28 pixels
hidden_neurons = 64
output_neurons = 10      # digits 0–9
lr = 0.1
epochs = 1000

# Weights and biases
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training the model
for i in range(epochs):
    # Forward Propagation
    hinp = np.dot(X_train, wh) + bh
    hlayer = relu(hinp)  # ReLU activation
    outinp = np.dot(hlayer, wout) + bout
    output = sigmoid(outinp)  # Sigmoid activation in output layer

    # Backpropagation
    error = y_train - output
    d_output = error * derivatives_sigmoid(output)

    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * derivatives_relu(hlayer)  # Derivative of ReLU

    # Update weights and biases
    wout += hlayer.T.dot(d_output) * lr
    wh += X_train.T.dot(d_hidden) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr  # Update bias
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr    # Update bias

    if i % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {i}, Loss: {loss:.4f}")

# Test the model
test_hidden = relu(np.dot(X_test, wh) + bh)
test_output = sigmoid(np.dot(test_hidden, wout) + bout)
predictions = np.argmax(test_output, axis=1)
actual = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == actual) * 100
print("Test Accuracy:", accuracy)
