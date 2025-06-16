import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Thiết lập giao diện
st.set_page_config(page_title="Kiểm Tra Văn Bản", page_icon="📝", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stTextArea textarea {border-radius: 8px; border: 1px solid #ccc;}
    .result-box {padding: 10px; border-radius: 8px; font-size: 18px;}
    .success {background-color: #e6f4e6; border: 1px solid #4CAF50;}
    .error {background-color: #f4e6e6; border: 1px solid #ff3333;}
    </style>
""", unsafe_allow_html=True)

st.title("📝 Kiểm Tra Văn Bản: Con Người hay AI?")
st.markdown("Nhập văn bản để kiểm tra xem nó được viết bởi con người hay tạo bởi AI.")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "transformer_model.keras",
            custom_objects={
                'positional_encoding': tf.keras.layers.Lambda(lambda x: x),
                'transformer_block': tf.keras.layers.Layer
            },
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Lỗi tải mô hình: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi tải tokenizer: {e}")
        return None

model = load_model()
tokenizer = load_tokenizer()

if not model or not tokenizer:
    st.stop()

# Nhập văn bản
user_input = st.text_area("Nhập văn bản cần kiểm tra", height=200, placeholder="Dán hoặc nhập văn bản tại đây...")

if st.button("🔍 Kiểm Tra"):
    if user_input.strip():
        max_len = 512
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        with st.spinner("Đang phân tích..."):
            prediction = model.predict(padded, verbose=0)
            result = "Do AI tạo" if prediction[0][0] > 0.5 else "Do con người viết"
            st.markdown(f"""
                <div class='result-box success'>
                    <strong>Kết quả:</strong> {result}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='result-box error'>
                Vui lòng nhập văn bản để kiểm tra!
            </div>
        """, unsafe_allow_html=True)