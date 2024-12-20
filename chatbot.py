from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import json
import pandas as pd
import random
import time
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer

import torch
from torch.utils.data import Dataset
# Tải các resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# dataset
filename = 'D:\\chatbot\\processed_bert_test_data.json'

# model
model_path = 'D:\\chatbot\\kaggle\\working\\chatbot'

def load_json_file(filename):
    with open(filename, encoding='utf-8') as f:
        file = json.load(f)
    return file

def create_df():
    return pd.DataFrame({'Pattern': [], 'Tag': []})

def extract_json_info(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            df.loc[len(df.index)] = [pattern, intent['tag']]
    return df

# Preprocessing
stemmer = PorterStemmer()
stopwords = [
    # Các từ phổ biến trong câu hỏi
    "là", "của", "và", "có", "cho", "đây", "rằng", "nhưng", "tôi", 
    "bạn", "với", "về", "thì", "lại", "này", "một", "nhiều", "nào",
    
    # Từ liên quan đến giáo dục/trường học
    "trường", "ngành", "học", "sinh viên", "đại học",
    "tuyển sinh", "đào tạo", "chương trình", "khoa",
    "năm", "hệ", "môn", "điểm", "thông tin", "giờ",
    
    # Cấu trúc câu hỏi thường gặp
    "như thế nào", "ra sao", "thế nào", "thì sao", "là gì",
    "ở đâu", "bao nhiêu", "khi nào", "làm sao", "bao giờ",
    
    # Từ để hỏi
    "ai", "gì", "nào", "đâu", "sao",
    
    # Các từ nối
    "mà", "các", "những", "từ", "tại", "theo", "sau", "sẽ", "vào",
    "do", "như", "hay", "còn", "bởi", "vì", "mình", "đến", "cũng",
    
    # Từ chỉ thời gian
    "hiện tại", "hiện nay", "đang", "trong", "nay", "khi",
    
    # Từ chỉ định
    "này", "đó", "kia", "ấy"
]
# Ký tự không cần thiết
ignore_words = [    '?', '!', ',', '.', ':', ';', '-', '', 
    '"', "'", '(', ')', '[', ']', '/', '\\',
    '+', '=', '@', '#', '$', '%', '^', '&', '*']

def preprocess_pattern(pattern):
    words = word_tokenize(pattern.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word not in ignore_words and word not in stopwords]
    return " ".join(stemmed_words)

# Load data
intents = load_json_file(filename)
df = create_df()
df = extract_json_info(intents, df)
df['Pattern'] = df['Pattern'].apply(preprocess_pattern)

# Prepare labels
labels = df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label = {id:label for id, label in enumerate(labels)}
label2id = {label:id for id, label in enumerate(labels)}

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

def predict_with_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = softmax(logits, dim=-1)
    pred_label_id = torch.argmax(probs, dim=-1).item()
    pred_label = model.config.id2label[pred_label_id]
    score = probs[0][pred_label_id].item()
    return {'label': pred_label, 'score': score}

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
            
            # Thực hiện dự đoán
            prediction = predict_with_score(user_input)
            score = prediction['score']
            predicted_label = prediction['label']

            # Xử lý response
            if score < 0.7:
                bot_response = "Xin lỗi, tôi không thể trả lời câu hỏi này."
            else:
                # Tìm intent tương ứng
                matching_intent = None
                for intent in intents['intents']:
                    if intent['tag'].strip() == predicted_label.strip():
                        matching_intent = intent
                        break

                if matching_intent:
                    bot_response = random.choice(matching_intent['responses'])
                else:
                    bot_response = "Xin lỗi, tôi không hiểu được câu hỏi của bạn."

            # Hiệu ứng typing
            typing_effect = ""
            for chunk in bot_response.split(" "):
                typing_effect += chunk + " "
                message_placeholder.markdown(typing_effect)
                time.sleep(0.05)

            # Final response
            message_placeholder.markdown(bot_response)

            # Add response to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": bot_response
            })


def main():
    chat_streamlit()

if __name__ == "__main__":
    main()
