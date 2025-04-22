import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from flask import Flask, render_template, request

app = Flask(__name__)

# === Activation Functions ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def derivatives_relu(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def xavier_initialization(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

# === Training Function ===
def train_model():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    y_train = y_train.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train)

    X, y = x_train[:800], y_train[:800]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    input_neurons = 784
    hidden_neurons = 64
    output_neurons = 10
    lr = 0.001
    epochs = 1000

    wh = xavier_initialization(input_neurons, hidden_neurons)
    bh = np.zeros((1, hidden_neurons))
    wout = xavier_initialization(hidden_neurons, output_neurons)
    bout = np.zeros((1, output_neurons))

    for i in range(epochs):
        hinp = np.dot(x_train, wh) + bh
        hlayer = relu(hinp)
        outinp = np.dot(hlayer, wout) + bout
        output = softmax(outinp)

        if np.isnan(output).any():
            print("NaN detected!")
            break

        error = y_train - output
        d_output = error * derivatives_sigmoid(output)
        error_hidden = d_output.dot(wout.T)
        d_hidden = error_hidden * derivatives_relu(hlayer)

        d_output = np.clip(d_output, -1.0, 1.0)
        d_hidden = np.clip(d_hidden, -1.0, 1.0)

        wout += hlayer.T.dot(d_output) * lr
        wh += x_train.T.dot(d_hidden) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr
        bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

        if i % 100 == 0:
            loss = np.mean(np.square(error))
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return wh, bh, wout, bout, encoder, x_train, y_train

# === Prediction Function ===
def predict_digit(X, wh, bh, wout, bout):
    hinp = np.dot(X, wh) + bh
    hlayer = relu(hinp)
    outinp = np.dot(hlayer, wout) + bout
    output = softmax(outinp)
    return np.argmax(output, axis=1)[0]

# === Web Route ===
@app.route("/")
def home():
    return "<h1>MNIST Digit Classifier Web App Running!</h1>"

# === Run Everything ===
if __name__ == "__main__":
    wh, bh, wout, bout, encoder, x_train, y_train = train_model()
    random_index = np.random.randint(0, x_train.shape[0])
    test_image = x_train[random_index].reshape(1, 784)
    true_label = y_train[random_index]
    predicted_digit = predict_digit(test_image, wh, bh, wout, bout)
    print(f"True Label: {np.argmax(true_label)}")
    print(f"Predicted Label: {predicted_digit}")

    # 👇 Start Flask app
    app.run(debug=True)
