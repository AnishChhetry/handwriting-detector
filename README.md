# ✍🏻 Handwriting Detector AI

A Flask-based web application that allows users to crowd-source a handwriting dataset and train a Convolutional Neural Network (CNN) directly from the browser to recognize characters.

The AI learns to recognize **62 classes**: Uppercase (A-Z), Lowercase (a-z), and Digits (0-9).

## ✨ Features

* **HTML5 Drawing Interface:** Draw characters directly in the browser using a mouse or touch screen.
* **Crowd-Sourced Dataset:** Every drawing is saved to a NumPy dataset (`.npy`), allowing the dataset to grow over time.
* **In-App Training:** No need for separate scripts. Click "Train Model" in the UI to retrain the neural network on the latest data immediately.
* **Real-Time Feedback:** Test the model in "Practice Mode" and get instant feedback on prediction confidence.
* **State-of-the-Art Architecture:** Uses a CNN (Convolutional Neural Network) with TensorFlow/Keras for high accuracy.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **ML/AI:** TensorFlow, Keras, NumPy
* **Frontend:** HTML5 Canvas, Bootstrap 5, JavaScript

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/AnishChhetry/handwriting-detector.git](https://github.com/AnishChhetry/handwriting-detector.git)
    cd handwriting-detector
    ```

2.  **Create a Virtual Environment** (Optional but recommended)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```
    Open your browser and go to `http://127.0.0.1:5000`

## 📖 How to Use

1.  **Add Data (Teach):**
    * Go to the "Add Data" tab.
    * The app will ask you to draw a specific letter (e.g., "G" or "q").
    * Draw it and click "Save". Do this for 10-20 characters to build a base dataset.

2.  **Train the Model:**
    * Click the yellow **⚡ Train Model** button in the navigation bar.
    * Wait for the spinner to finish. The app is now retraining the neural network on your new drawings.

3.  **Practice (Test):**
    * Go to the "Practice" tab.
    * Draw any character.
    * Click "Check" to see if the AI can recognize it!

## 📂 Project Structure

```text
handwriting-detector/
├── app.py              # Main Flask application and training logic
├── data/               # Stores the .npy datasets (created automatically)
├── templates/          # HTML templates (Bootstrap 5)
│   ├── base.html
│   ├── index.html
│   ├── addData.html
│   └── practice.html
├── letter.keras        # The trained AI model (generated after training)
├── requirements.txt    # Python dependencies
└── README.md