🧠 MNIST Digit Classifier Web App
This is a simple web application that uses a neural network built from scratch (using only NumPy) to classify handwritten digits (0–9) from the MNIST dataset. The model is integrated into a web app using Flask.

📌 Features
Built a feedforward neural network using only NumPy

Trained on the MNIST dataset with ReLU and softmax activation

Used Xavier initialization for better weight distribution

Real-time digit prediction through the Flask web interface

Clean HTML form to upload digit images

🛠️ Technologies Used
Python

NumPy

Scikit-learn (for OneHotEncoder and train/test split)

TensorFlow (only to load MNIST dataset)

Flask (to build the web interface)

HTML (for the front-end upload form)

🧪 How It Works
Data Loading & Preprocessing:

MNIST dataset is loaded using tensorflow.keras.datasets.

Images are flattened from 28x28 to 784.

Labels are one-hot encoded.

Neural Network Architecture:

Input Layer: 784 neurons

Hidden Layer: 64 neurons (ReLU activation)

Output Layer: 10 neurons (Softmax activation)

Weights initialized using Xavier Initialization

Training:

1000 epochs using backpropagation and gradient descent.

Loss printed every 100 epochs for monitoring.

Prediction:

Given an uploaded digit image, it is preprocessed and passed through the trained model.

Predicted digit is shown on the web page.

🚀 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/mnist-digit-classifier.git
cd mnist-digit-classifier
2. Install dependencies
Make sure you have Python installed. Then install required packages:

bash
Copy
Edit
pip install numpy scikit-learn tensorflow flask
3. Run the application
bash
Copy
Edit
python app.py
4. Visit in browser
Open your browser and go to:
http://127.0.0.1:5000

📷 Sample HTML Form
html
Copy
Edit
<form action="/predict" method="POST" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required>
    <input type="submit" value="Submit">
</form>
🔮 Example Prediction
After training, the app will randomly select a test image and show:

mathematica
Copy
Edit
True Label: 7
Predicted Label: 7
📁 Project Structure
php
Copy
Edit
├── app.py              # Main Flask app + training + prediction logic
├── templates/
│   └── index.html      # HTML template for upload form
├── static/             # (Optional) for CSS/images if needed
└── README.md           # You're reading this!
🤝 Contribution
Feel free to fork this project and improve it! You can add:

Handwritten digit image preprocessing

Drag and draw digit pad

Live model retraining

📜 License
This project is open-source and free to use for educational purposes.

