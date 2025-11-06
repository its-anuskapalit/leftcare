# 🌿 AI-Powered Plant Disease Detection Web App

This project is a Flask-based web application that identifies plant diseases from leaf images using a trained deep learning model. Users can upload an image, and the app will predict whether the plant is healthy or affected by a disease, along with possible remedies.

---

## 🧠 Project Structure

📁 project-folder/
├── app.py # Main Flask application
├── chatbot.py # Chatbot logic for plant care queries
├── inference.py # Model inference and prediction script
├── model.py # Model loading and preprocessing utilities
├── templates/
│ └── index.html # Frontend HTML interface
├── 1.jpeg, 2.jpeg, 3.jpeg # Sample test images
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 Features

- 🌱 Upload plant leaf images for disease detection  
- 💬 Integrated chatbot for plant care tips and remedies  
- 🧩 Uses TensorFlow Lite model for lightweight inference  
- 🖼️ Simple and responsive HTML interface  
- 🔍 Provides suggestions and preventive measures for each disease  

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **AI/ML:** TensorFlow Lite / OpenCV / NumPy  
- **Frontend:** HTML, CSS, JavaScript  
- **Model:** Pre-trained CNN for leaf disease classification  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
python app.py
Open in browser

cpp
Copy code
http://127.0.0.1:5000/
📸 Sample Images
Healthy Leaf	Infected Leaf	Prediction Output

🤖 Chatbot Commands
You can ask the chatbot questions like:

“How often should I water my tomatoes?”

“What should I do if my plant has yellow leaves?”

“How to treat powdery mildew naturally?”

“I’m going on vacation — how can I keep my plants alive?”

📂 Model Integration
The model is loaded via TensorFlow Lite (.tflite) for optimized performance:

python
Copy code
interpreter = tf.lite.Interpreter(model_path="models/leafcare_model.tflite")
interpreter.allocate_tensors()
Predictions are processed in inference.py and displayed on the web interface.

📈 Future Improvements
Add multi-language chatbot support

Include real-time camera capture

Expand dataset for more plant species

Deploy on cloud (Render / AWS / Hugging Face Spaces)

👨‍💻 Author
Anuska Palit
🌐 LinkedIn | 🧠 AI Research | 🌾 Sustainable Tech Innovator

🪴 License
This project is licensed under the MIT License – feel free to use, modify, and share.

“AI for a greener planet — because every leaf deserves care.” 🍃
