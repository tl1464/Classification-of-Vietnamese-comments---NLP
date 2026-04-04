import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
HF_MODEL_ID = "t59thang/hmvpt-toxic"
HF_DATASET_ID = "t59thang/hmvpt-data"

LABEL_MAP = {
    0: ("✅ Đây là bình luận Tích cực ", "green"),
    1: ("☠️ Đây là bình luận Tiêu cực", "red"),
    2: ("🚫 Đây là bình luận Spam", "orange")
}

st.set_page_config(page_title="AI Toxic Comment Detection", layout="wide")

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model

# =========================
# LOAD DATA (CACHE)
# =========================
@st.cache_data
def load_data():
    dataset = load_dataset(HF_DATASET_ID, split="train")
    return dataset.to_pandas()

# =========================
# PREDICT
# =========================
def predict(text, tokenizer, model):
    inputs = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label = torch.argmax(probs).item()

    return label, probs[label].item()

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio("📌 Điều hướng", [
    "1. Giới thiệu & EDA",
    "2. Triển khai mô hình",
    "3. Đánh giá & Hiệu năng"
])

df = load_data()
tokenizer, model = load_model()

# =========================
# PAGE 1 - EDA
# =========================
if page == "1. Giới thiệu & EDA":

    st.title("📊 Phân tích bình luận độc hại (Toxic Comment Detection)")

    st.markdown("""
    **👤 Sinh viên: ĐỖ ĐOÀN MINH THẮNG - 22T1020424 

    ### 🎯 Mục tiêu
    Xây dựng hệ thống AI giúp phát hiện bình luận độc hại, spam trên mạng xã hội.

    ### 💡 Giá trị thực tiễn
    - Hỗ trợ kiểm duyệt nội dung tự động  
    - Giảm toxic online  
    - Ứng dụng trong Facebook, TikTok, diễn đàn  
    """)

    st.subheader("📄 Dữ liệu mẫu")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)

    # 📊 Biểu đồ 1
    with col1:
        st.subheader("📊 Phân phối nhãn")
        fig, ax = plt.subplots()
        df['label'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(["Đây là bình luận Tích cực", "Đây là bình luận Tiêu cực", "Đây là bình luận Spam"], rotation=0)
        st.pyplot(fig)

    # 📊 Biểu đồ 2
    with col2:
        st.subheader("📊 Độ dài bình luận")
        df['length'] = df['comment'].apply(len)
        fig, ax = plt.subplots()
        df['length'].hist(bins=30, ax=ax)
        st.pyplot(fig)

    st.markdown("""
    ### 📌 Nhận xét:
    - Dữ liệu có thể bị **lệch giữa các nhãn**
    - Bình luận ngắn chiếm đa số
    - Các từ toxic thường xuất hiện rõ ràng
    """)

# =========================
# PAGE 2 - MODEL
# =========================
elif page == "2. Triển khai mô hình":

    st.title("🤖 Dự đoán bình luận")

    text = st.text_area("✍️ Nhập bình luận:", height=120)

    if st.button("🔍 Phân tích"):
        if text.strip():
            label, confidence = predict(text, tokenizer, model)
            label_text, color = LABEL_MAP[label]

            st.markdown(f"### Kết quả: :{color}[{label_text}]")
            st.progress(confidence, text=f"Độ tin cậy: {confidence:.2%}")

    st.divider()

    st.subheader("⚡ Test nhanh")
    
    sample = st.selectbox(
        "Chọn câu mẫu",
        [
            "công an đánh dân",
            "đồ ngu như chó",
            "spam bán hàng",
            "hôm nay trời đẹp"
        ]
    )

    if st.button("🚀 Dự đoán câu mẫu"):
        label, confidence = predict(sample, tokenizer, model)
        label_text, color = LABEL_MAP[label]

        st.write(sample)
        st.markdown(f"### :{color}[{label_text}]")
        st.write(f"Confidence: {confidence:.2%}")

# =========================
# PAGE 3 - EVALUATION
# =========================
elif page == "3. Đánh giá & Hiệu năng":

    st.title("📈 Đánh giá mô hình")

    sample_df = df.sample(200)

    y_true = sample_df['label'].tolist()
    y_pred = []

    for text in sample_df['comment']:
        label, _ = predict(text, tokenizer, model)
        y_pred.append(label)

    report = classification_report(y_true, y_pred, output_dict=True)

    st.subheader("📊 Metrics")
    st.json(report)

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### 📌 Phân tích lỗi:
    - Model dễ nhầm giữa **toxic và spam**
    - Một số câu trung tính nhưng có từ nhạy cảm bị phân loại sai
    - Cần thêm dữ liệu để cải thiện
    """)