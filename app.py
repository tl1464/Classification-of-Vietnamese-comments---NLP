import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL_ID = "t59thang/hmvpt-toxic"

LABEL_MAP = {
    0: ("✅ SẠCH", "green"),
    1: ("☠️ ĐỘC HẠI", "red"),
    2: ("🚫 SPAM", "orange")
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    processed = str(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    label = torch.argmax(probs).item()
    return label, probs[label].item()

st.title("🛡️ Phát hiện bình luận độc hại")

tokenizer, model = load_model()

text = st.text_area("Nhập comment:")

if st.button("Phân tích"):
    if text:
        label, conf = predict(text, tokenizer, model)
        st.write(LABEL_MAP[label][0])
        st.write(f"Độ tin cậy: {conf:.2%}")