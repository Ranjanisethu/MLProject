import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def derivatives_relu(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # Numerical stability for softmax
    exp_x = np.exp(x - np.max(x))  # Subtract max value to prevent overflow
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Initialize weights using Xavier initialization
def xavier_initialization(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

def train_model():
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

    # Network parameters
    input_neurons = 784      # 28x28 pixels
    hidden_neurons = 64
    output_neurons = 10      # digits 0–9
    lr = 0.001  # Learning rate
    epochs = 1000

    # Initialize weights using Xavier initialization
    wh = xavier_initialization(input_neurons, hidden_neurons)
    bh = np.zeros((1, hidden_neurons))  # Bias initialized to 0
    wout = xavier_initialization(hidden_neurons, output_neurons)
    bout = np.zeros((1, output_neurons))  # Bias initialized to 0

    # Training the model
    for i in range(epochs):
        # Forward Propagation
        hinp = np.dot(X_train, wh) + bh
        hlayer = relu(hinp)  # ReLU activation
        outinp = np.dot(hlayer, wout) + bout
        output = softmax(outinp)  # Softmax activation in output layer

        # Check for NaN values
        if np.isnan(output).any():
            print("NaN detected in output!")
            break

        # Backpropagation
        error = y_train - output
        d_output = error * derivatives_sigmoid(output)

        error_hidden = d_output.dot(wout.T)
        d_hidden = error_hidden * derivatives_relu(hlayer)  # Derivative of ReLU

        # Clip gradients to prevent exploding gradients
        gradient_clip_value = 1.0
        d_output = np.clip(d_output, -gradient_clip_value, gradient_clip_value)
        d_hidden = np.clip(d_hidden, -gradient_clip_value, gradient_clip_value)

        # Update weights and biases
        wout += hlayer.T.dot(d_output) * lr
        wh += X_train.T.dot(d_hidden) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr  # Update bias
        bh += np.sum(d_hidden, axis=0, keepdims=True) * lr    # Update bias

        if i % 100 == 0:
            loss = np.mean(np.square(error))
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return wh, bh, wout, bout, encoder

# Predict digit function
def predict_digit(X, wh, bh, wout, bout):
    """
    Function to predict the digit for a given image.
    X: Input image array (after preprocessing)
    wh, bh: Weights and biases for the hidden layer
    wout, bout: Weights and biases for the output layer
    """
    # Forward pass for prediction
    hinp = np.dot(X, wh) + bh
    hlayer = relu(hinp)  # ReLU activation
    outinp = np.dot(hlayer, wout) + bout
    output = softmax(outinp)  # Softmax activation for output

    # Get the predicted digit
    prediction = np.argmax(output, axis=1)  # Returns the index with highest probability
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Train the model
    wh, bh, wout, bout, encoder = train_model()

    # Test the prediction
    test_image = np.random.rand(1, 784)  # Example random image (flattened)
    predicted_digit = predict_digit(test_image, wh, bh, wout, bout)
    print(f"Predicted digit: {predicted_digit}")
