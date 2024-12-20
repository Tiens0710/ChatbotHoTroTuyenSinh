import streamlit as st
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import json
import pandas as pd
import time
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Tải NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Dataset và model paths
filename = 'processed_bert_test_data.json'
model_path = 'kaggle/working/chatbot'

# Preprocessing setup
stemmer = PorterStemmer()
stopwords = [
    # Các từ phổ biến trong câu hỏi
    "là", "của", "và", "có", "cho", "đây", "rằng", "nhưng", "tôi", 
    "bạn", "với", "về", "thì", "lại", "này", "một", "nhiều", "nào",
    # ... (giữ nguyên danh sách stopwords của bạn)
]

ignore_words = ['?', '!', ',', '.', ':', ';', '-', '', 
    '"', "'", '(', ')', '[', ']', '/', '\\',
    '+', '=', '@', '#', '$', '%', '^', '&', '*']

class ChatbotCore:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intents = None
        self.initialize()

    def initialize(self):
        """Khởi tạo model và resources"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.intents = self.load_json_file(filename)
            print("Model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def load_json_file(self, filename):
        with open(filename, encoding='utf-8') as f:
            return json.load(f)

    def preprocess_pattern(self, pattern):
        words = word_tokenize(pattern.lower())
        stemmed_words = [stemmer.stem(word) for word in words 
                        if word not in ignore_words and word not in stopwords]
        return " ".join(stemmed_words)

    def predict_with_score(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = softmax(logits, dim=-1)
        pred_label_id = torch.argmax(probs, dim=-1).item()
        pred_label = self.model.config.id2label[pred_label_id]
        score = probs[0][pred_label_id].item()
        return {'label': pred_label, 'score': score}

    def get_response(self, text):
        """Process input and return response"""
        try:
            # Preprocess and predict
            processed_text = self.preprocess_pattern(text)
            prediction = self.predict_with_score(processed_text)
            score = prediction['score']
            predicted_label = prediction['label']

            # Generate response
            if score < 0.7:
                return {
                    'response': "Xin lỗi, tôi không thể trả lời câu hỏi này.",
                    'confidence': score,
                    'intent': 'unknown'
                }

            # Find matching intent
            for intent in self.intents['intents']:
                if intent['tag'].strip() == predicted_label.strip():
                    return {
                        'response': random.choice(intent['responses']),
                        'confidence': score,
                        'intent': predicted_label
                    }

            return {
                'response': "Xin lỗi, tôi không hiểu được câu hỏi của bạn.",
                'confidence': score,
                'intent': 'unknown'
            }

        except Exception as e:
            print(f"Error processing request: {e}")
            return {
                'response': "Đã xảy ra lỗi trong quá trình xử lý.",
                'confidence': 0,
                'intent': 'error'
            }

# Initialize chatbot
chatbot = ChatbotCore()

# Flask routes
@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message parameter'}), 400

        response = chatbot.get_response(data['message'])
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

# Streamlit interface
def chat_streamlit():
    st.title("🤖 Hỗ Trợ Sinh Viên")
    st.write("Tôi có thể giúp gì cho bạn")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input area
    if user_input := st.chat_input("Your message:"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Chatbot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Get response from chatbot
            response_data = chatbot.get_response(user_input)
            bot_response = response_data['response']

            # Typing effect
            typing_effect = ""
            for chunk in bot_response.split(" "):
                typing_effect += chunk + " "
                message_placeholder.markdown(typing_effect)
                time.sleep(0.05)

            # Final response
            message_placeholder.markdown(bot_response)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": bot_response
            })

def run_flask():
    """Run Flask server"""
    app.run(host='0.0.0.0', port=5000)

def main():
    """Main function to run both Streamlit and Flask"""
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # This ensures the thread will die when the main program exits
    flask_thread.start()
    
    # Run Streamlit interface
    chat_streamlit()

if __name__ == "__main__":
    main()
