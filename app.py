import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------ Load dan Compile Model ------------------ #
@st.cache_resource
def load_model():
    from keras.layers import InputLayer  # Tambahkan ini
    model = tf.keras.models.load_model('best_model.h5', custom_objects={'InputLayer': InputLayer})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = load_model()
class_names = ['busuk', 'matang', 'mentah', 'segar']

# ------------------ Konfigurasi Halaman ------------------ #
st.set_page_config(page_title="Deteksi Kesegaran Pisang", layout="centered")
st.title("🍌 Deteksi Tingkat Kesegaran Buah Pisang")
st.markdown("""
Aplikasi ini merupakan bagian dari penelitian skripsi berjudul:

> **_Deteksi Tingkat Kesegaran Buah Pisang sebagai Upaya Pencegahan Risiko Kesehatan Menggunakan CNN Berbasis Website_**

Silakan unggah gambar atau gunakan kamera untuk mendeteksi tingkat kesegaran buah pisang secara otomatis.
""")

# ------------------ Inisialisasi Session State ------------------ #
if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = False

# ------------------ Input: Tombol Kamera dan Upload Gambar ------------------ #
col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Gunakan Kamera"):
        st.session_state.camera_mode = True
with col2:
    if st.button("📁 Gunakan Upload"):
        st.session_state.camera_mode = False

uploaded_file = None
camera_image = None
img = None

if st.session_state.camera_mode:
    camera_image = st.camera_input("Ambil gambar pisang menggunakan kamera")
else:
    uploaded_file = st.file_uploader("Unggah gambar pisang", type=["jpg", "jpeg", "png"])

# ------------------ Fungsi Prediksi ------------------ #
def predict(image: Image.Image):
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 150, 150, 3))
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_names[class_index], float(prediction[0][class_index])

# ------------------ Penjelasan Hasil ------------------ #
def get_explanation(label: str) -> str:
    explanations = {
        "busuk": (
            "⚠️ **Pisang Busuk**\n\n"
            "- Tidak layak konsumsi.\n"
            "- Berpotensi menyebabkan gangguan kesehatan akibat kontaminasi mikroorganisme.\n\n"
            "**Status Kesehatan: ❌ Tidak Sehat**"
        ),
        "matang": (
            "🍌 **Pisang Matang**\n\n"
            "- Siap konsumsi dan kaya nutrisi.\n"
            "- Kandungan serat dan kalium dalam kondisi optimal.\n\n"
            "**Status Kesehatan: ✅ Sehat**"
        ),
        "mentah": (
            "📉 **Pisang Mentah**\n\n"
            "- Belum layak dikonsumsi langsung.\n"
            "- Disarankan untuk disimpan hingga matang.\n\n"
            "**Status Kesehatan: ⚠️ Belum Layak Konsumsi**"
        ),
        "segar": (
            "🌿 **Pisang Segar**\n\n"
            "- Kondisi baik dan segar.\n"
            "- Cocok untuk penyimpanan jangka pendek atau distribusi.\n\n"
            "**Status Kesehatan: ✅ Sehat**"
        )
    }
    return explanations.get(label, "❓ Informasi label tidak tersedia.")

# ------------------ Deteksi dan Output ------------------ #
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📁 Gambar yang Diupload", use_container_width=True)
elif camera_image:
    img = Image.open(camera_image)
    st.image(img, caption="📸 Gambar dari Kamera", use_container_width=True)

if img is not None:
    label, confidence = predict(img)
    st.subheader("📊 Hasil Deteksi")
    st.success(f"**Kategori Kesegaran**: `{label.upper()}`")
    st.progress(int(confidence * 100))
    st.caption(f"🎯 Tingkat Keyakinan Model: `{confidence*100:.2f}%`")
    
    st.info(get_explanation(label))

    # Penjelasan tambahan untuk mendukung judul skripsi
    if label == "busuk":
        st.error("🚫 **Rekomendasi:** Segera buang pisang ini. Deteksi ini merupakan langkah pencegahan risiko kesehatan akibat kontaminasi bakteri atau jamur.")
        st.markdown("> 🔍 *Deteksi dini terhadap pisang busuk mendukung upaya preventif terhadap potensi keracunan makanan.*")
    elif label == "matang":
        st.success("👍 **Rekomendasi:** Pisang dalam kondisi ideal untuk dikonsumsi. Aman dan bernutrisi tinggi.")
        st.markdown("> 📌 *Konsumen disarankan mengonsumsi pisang pada fase ini untuk mendapatkan manfaat kesehatan optimal.*")
    elif label == "mentah":
        st.warning("⚠️ **Rekomendasi:** Pisang belum siap konsumsi. Simpan terlebih dahulu hingga matang.")
        st.markdown("> 🕒 *Penundaan konsumsi adalah bentuk pencegahan dari gangguan pencernaan akibat pati yang belum terurai.*")
    elif label == "segar":
        st.info("📦 **Rekomendasi:** Cocok untuk distribusi atau stok penyimpanan. Segar dan belum terlalu matang.")
        st.markdown("> 📈 *Kelayakan distribusi dalam rantai pasok buah segar dapat ditingkatkan dengan deteksi kategori ini.*")

# ------------------ Footer ------------------ #
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 - Aplikasi Deteksi Kesegaran Buah Pisang<br>by <strong>Arya Gunawan</strong>"
    "</div>",
    unsafe_allow_html=True
)