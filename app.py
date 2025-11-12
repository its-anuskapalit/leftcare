<<<<<<< HEAD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import base64
import io
import numpy as np
from PIL import Image

# Import core logic components
from inference import predict_leaf
from chatbot import get_chatbot_response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#Utility Functions 

def save_uploaded_image(file):
    if file:
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

def encode_heatmap_to_base64(heatmap_array: np.array) -> str:
    img_data = (heatmap_array * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L') 
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'leaf_image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['leaf_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        temp_path = None
        try:
            temp_path = save_uploaded_image(file)
            results = predict_leaf(temp_path)
            heatmap_b64 = encode_heatmap_to_base64(results['attention_map'])
            response_data = {
                'status': 'success',
                'original_filename': file.filename,
                'prediction': {
                    'label': results['label'],
                    'confidence': f"{results['confidence']:.2f}%",
                    'health_index': f"{results['health_index']:.2f}%",
                    'remedies': results['remedies']
                },
                'heatmap_overlay': heatmap_b64 
            }
            return jsonify(response_data)
        except FileNotFoundError as e:
            print(e)
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot queries via POST request."""
    data = request.get_json()
    user_query = data.get('message', '').strip()
    
    if not user_query:
        return jsonify({'response': 'Please provide a message.'})

    try:
        response_text = get_chatbot_response(user_query)
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify({'response': 'Sorry, the chatbot service is currently down.'}), 500

if __name__ == '__main__':
    print("--- Starting LeafCare Flask Application ---")
    print("NOTE: ML components are mocked. Install dependencies for full functionality.")
    # In production/deployment, host and port should be configured securely
=======
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import base64
import io
import numpy as np
from PIL import Image

# Import core logic components
from inference import predict_leaf
from chatbot import get_chatbot_response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#Utility Functions 

def save_uploaded_image(file):
    if file:
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

def encode_heatmap_to_base64(heatmap_array: np.array) -> str:
    img_data = (heatmap_array * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L') 
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'leaf_image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['leaf_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        temp_path = None
        try:
            temp_path = save_uploaded_image(file)
            results = predict_leaf(temp_path)
            heatmap_b64 = encode_heatmap_to_base64(results['attention_map'])
            response_data = {
                'status': 'success',
                'original_filename': file.filename,
                'prediction': {
                    'label': results['label'],
                    'confidence': f"{results['confidence']:.2f}%",
                    'health_index': f"{results['health_index']:.2f}%",
                    'remedies': results['remedies']
                },
                'heatmap_overlay': heatmap_b64 
            }
            return jsonify(response_data)
        except FileNotFoundError as e:
            print(e)
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot queries via POST request."""
    data = request.get_json()
    user_query = data.get('message', '').strip()
    
    if not user_query:
        return jsonify({'response': 'Please provide a message.'})

    try:
        response_text = get_chatbot_response(user_query)
        return jsonify({'response': response_text})
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify({'response': 'Sorry, the chatbot service is currently down.'}), 500

if __name__ == '__main__':
    print("--- Starting LeafCare Flask Application ---")
    print("NOTE: ML components are mocked. Install dependencies for full functionality.")
    # In production/deployment, host and port should be configured securely
>>>>>>> 97afc2203075cca1880a1ca8a74f6f5d337959e5
    app.run(debug=True)