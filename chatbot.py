import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
FAQ_PATH = 'data/faq.json'
FALLBACK_THRESHOLD = 0.75 # Cosine Similarity Threshold
MODEL_NAME = 'all-MiniLM-L6-v2' # Efficient and effective transformer model

try:
    SIM_MODEL = SentenceTransformer(MODEL_NAME)
    
    with open(FAQ_PATH, 'r') as f:
        FAQ_DATA = json.load(f)
    QA_PAIRS = [(item['question'], item['answer']) for item in FAQ_DATA]
    STORED_QUESTIONS = [pair[0] for pair in QA_PAIRS]
    STORED_EMBEDDINGS = SIM_MODEL.encode(STORED_QUESTIONS, convert_to_tensor=True)
    
except Exception as e:
    print(f"Chatbot initialization failed: {e}")
    SIM_MODEL = None
    QA_PAIRS = []

def get_chatbot_response(user_query: str) -> str:
    if SIM_MODEL is None or not QA_PAIRS:
        return "The plant care knowledge base is currently unavailable."

    # 1. Encode user query
    query_embedding = SIM_MODEL.encode([user_query], convert_to_tensor=True)
    
    # 2. Calculate similarity
    similarities = cosine_similarity(query_embedding.cpu(), STORED_EMBEDDINGS.cpu())[0]
    
    # 3. Find best match
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]
    
    if best_similarity >= FALLBACK_THRESHOLD:
        # Found a relevant answer
        return QA_PAIRS[best_match_index][1]
    else:
        # Fallback as per requirement
        return "I’m not sure about that specific query. For detailed issues, please consult a local gardening expert."