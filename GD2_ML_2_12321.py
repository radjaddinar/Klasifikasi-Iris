import os
import pickle
import numpy as np
import streamlit as st

# === 1) Load Model ===
model_path = os.path.join(os.path.dirname(__file__), "RF_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# === 2) Judul Aplikasi ===
st.title("Klasifikasi Iris - Random Forest Classifier")
st.title("Dinarradja Raditya Dharma - 230712321")  # isikan dengan nama dan NPM praktikan

st.write("""
Masukkan panjang dan lebar petal (kelopak bunga) untuk memprediksi jenis bunga:
- **0 = Setosa**
- **1 = Versicolor**
- **2 = Virginica**
""")

# === 3) Input User ===
petal_length = st.number_input("Petal Length [cm]", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.number_input("Petal Width [cm]", min_value=0.0, max_value=10.0, value=1.3, step=0.1)

# === 4) Prediksi ===
if st.button("Prediksi"):
    # Buat array data baru
    X_new = np.array([[petal_length, petal_width]])
    
    # Lakukan prediksi
    y_pred = model.predict(X_new)
    
    # Tentukan label hasil prediksi
    if y_pred[0] == 0:
        label = "Setosa (0)"
    elif y_pred[0] == 1:
        label = "Versicolor (1)"
    else:
        label = "Virginica (2)"
    
    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{label}**")
