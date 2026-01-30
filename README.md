# Brain Tumor Detection MRI 🧠

Proyek ini adalah aplikasi berbasis Deep Learning untuk mendeteksi tumor otak melalui citra MRI menggunakan model Keras/TensorFlow.

## Fitur
* Deteksi tumor otak otomatis dari file gambar MRI.
* Model menggunakan **Focal Loss** untuk menangani ketidakseimbangan data (Imbalanced Data).
* Antarmuka sederhana (Sebutkan jika Anda menggunakan Streamlit/Flask/Gradio).

## Cara Menjalankan Aplikasi

### 1. Clone Repositori
Karena proyek ini menggunakan **Git LFS** untuk menyimpan model sebesar 250MB+, pastikan Anda sudah menginstal [Git LFS](https://git-lfs.github.com/) di komputer Anda, lalu jalankan:

```bash
git clone [https://github.com/satriaaawan/braintumor_detection.git](https://github.com/satriaaawan/braintumor_detection.git)
cd braintumor_detection
git lfs pull

python main.py / streamlit run app.py
