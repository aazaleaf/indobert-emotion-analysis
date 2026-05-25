import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Menggunakan cache agar model tidak di-load berulang kali setiap ada interaksi
@st.cache_resource
def load_model():
    # Nama repo SUDAH DIPERBAIKI sesuai dengan yang ada di Hugging Face kamu
    model_path = "aazaleaf/indobert-emosi-prdect"
    
    # Streamlit akan otomatis mendownload model dari Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# ---> INI BARIS YANG KURANG SEBELUMNYA <---
# Memanggil fungsi agar tokenizer dan model benar-benar dibuat
tokenizer, model = load_model()

# Bagian UI Streamlit
st.title("Klasifikasi Emosi Review Produk 📝")
st.write("Masukkan ulasan untuk mengetahui emosi dari teks tersebut.")

user_input = st.text_area("Tulis ulasan di sini:")

if st.button("Prediksi"):
    if user_input:
        # Proses tokenisasi (sesuaikan dengan kode preprocessing di notebook-mu)
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # Prediksi dengan model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mengambil hasil prediksi
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions).item()
        
        # Tampilkan hasil (Note: ubah angka ke nama label sesuai dataset PRDECT-ID)
        st.success(f"Hasil Klasifikasi: Kelas {predicted_class}")
    else:
        st.warning("Teks tidak boleh kosong!")