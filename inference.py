import numpy as np
import tensorflow as tf
import cv2
import json
import os
MODEL_PATH_TFLITE = 'models/leafcare_model.tflite'
REMEDIES_PATH = 'data/remedies.json'
CLASS_LABELS = [
    'Healthy', 
    'Late Blight', 
    'Powdery Mildew', 
    'Spider Mites', 
    'Aphids', 
    'Bacterial Spot', 
    'Leaf Spot', 
    'Rust', 
    'Blight (Early Blight)', 
    'Root Rot', 
    'Fusarium Wilt', 
    'Nutrient Deficiency (General)',
    'Nitrogen Deficiency',
    'Iron Chlorosis',
    'Mosaic Virus',
    'Thrips',
    'Whiteflies',
    'Scale Insects'
]
INPUT_SIZE = (224, 224)

try:
    with open(REMEDIES_PATH, 'r') as f:
        REMEDIES_DB = json.load(f)
except FileNotFoundError:
    print(f"Error: Remedies database not found at {REMEDIES_PATH}")
    REMEDIES_DB = {}

def load_model_and_interpreter():
    if not os.path.exists(MODEL_PATH_TFLITE):
        print(f"Warning: TFLite model not found at {MODEL_PATH_TFLITE}. Using synthetic data.")
        return None
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH_TFLITE)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def generate_synthetic_heatmap(H, W):
    heatmap = np.zeros((H, W), dtype=np.float32)
    center_y, center_x = int(H * 0.4), int(W * 0.6)
    radius = min(H, W) // 5
    for y in range(H):
        for x in range(W):
            dist_sq = (y - center_y)**2 + (x - center_x)**2
            if dist_sq < radius**2:
                heatmap[y, x] = 1.0 - (np.sqrt(dist_sq) / radius)
                
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    return heatmap

def predict_leaf(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    original_shape = img.shape 
    H, W, _ = original_shape 
    input_tensor = cv2.resize(img, INPUT_SIZE)
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    mock_class_logits = np.random.randn(1, len(CLASS_LABELS)) 
    mock_classes = tf.nn.softmax(mock_class_logits).numpy()[0]
    mock_health_raw = np.random.rand(1, 1) 
    health_index = float(mock_health_raw[0, 0] * 100.0)
    mock_attention_map = generate_synthetic_heatmap(H, W) 
    predicted_idx = np.argmax(mock_classes)
    label = CLASS_LABELS[predicted_idx]
    confidence = float(mock_classes[predicted_idx] * 100.0) 
    health_index_final = round(max(0.0, min(100.0, health_index)), 2)
    remedies = REMEDIES_DB.get(label, ["No specific remedy found. Check general care instructions."])
    if confidence < 70.0:
        # Override the label and remedies to simulate the fallback logic
        label = f"{label} (Uncertain - Prototypical Fallback)" 
        remedies.insert(0, "Prediction confidence is low. Consult the chatbot or a specialist for confirmation.")
    
    return {
        'label': label,
        'confidence': round(confidence, 2),
        'health_index': health_index_final,
        'attention_map': mock_attention_map, 
        'remedies': remedies
    }