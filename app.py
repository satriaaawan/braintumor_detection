import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# --------- FOCAL LOSS (Custom Function) ----------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# --------- LABELS (6 Kelas) ----------
labels = ['astrocitoma','glioma', 'meningioma', 'neurocitoma', 'notumor', 'pituitary',]

# --------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Deteksi Tumor Otak", page_icon="🧠", layout="centered")
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["🏠 Home", "📤 Upload MRI"])

# --------- LOAD MODEL (.h5) ----------
@st.cache_resource
def load_brain_model():
    return load_model("model_focal2.h5", custom_objects={"focal_loss_fixed": focal_loss()})

model = load_brain_model()

# --------- HOME PAGE ----------
if page == "🏠 Home":
    st.title("🧠 Rumah Sakit Dio")
    st.markdown("""
    ### Selamat datang!
    Aplikasi ini memanfaatkan deep learning untuk mendeteksi **jenis tumor otak** dari hasil scan MRI.
    
    #### Jenis Tumor:
    - Astrocitoma
    - Glioma
    - Meningioma
    - Neurocitoma
    - Pituitary
    - Tidak Ada Tumor
    """)

# --------- UPLOAD PAGE ----------
elif page == "📤 Upload MRI":
    st.title("📤 Upload Gambar MRI")
    st.write("Silakan upload gambar MRI otak untuk mendeteksi jenis tumor.")

    uploaded_file = st.file_uploader("Pilih file gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🧾 Pratinjau Gambar MRI", use_container_width=True)

        # Preprocessing
        img = image.resize((299, 299))
        img = img.convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 299, 299, 3)

        if st.button("🔎 Prediksi"):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = labels[predicted_class]
            confidence = prediction[0][predicted_class] * 100

            st.success(f"🧬 Jenis Tumor Terdeteksi: **{predicted_label.upper()}**")
            st.markdown(f"### 🎯 Akurasi Prediksi: `{confidence:.2f}%`")

            # Tampilkan hasil probabilitas sebagai tabel
            st.markdown("### 📊 Probabilitas Kelas:")
            df_probs = pd.DataFrame({
                "Kelas": labels,
                "Probabilitas (%)": [f"{p * 100:.2f}%" for p in prediction[0]]
            })

            st.dataframe(df_probs, use_container_width=True)
