import streamlit as st
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os
import pandas as pd

# --- 1️⃣ Atur halaman agar full width ---
# Mengatur halaman untuk menggunakan layout lebar penuh
st.set_page_config(page_title="DL-CIS", page_icon="🎨", layout="wide")

# --- 2️⃣ Tambahkan Header dan Footer dengan CSS ---
# Menambahkan style untuk header dan footer agar tetap di atas dan bawah dengan lebar penuh
st.markdown(
    """
    <style>
    /* ===================== Default (Light Mode) ===================== */
    :root {
        color-scheme: light;
        --header-bg: #6D2323;
        --header-text: white;
        --body-bg: #FEF9E1;
        --body-text: #000000;
        --sidebar-bg: #FEF9E1;
        --sidebar-border: #A31D1D;
        --footer-bg: #6D2323;
        --footer-text: white;
    }

    /* ===================== Dark Mode Override ===================== */
    @media (prefers-color-scheme: dark) {
        :root {
            --header-bg: #1e1e1e;
            --header-text: #ffffff;
            --body-bg: #121212;
            --body-text: #ffffff;
            --sidebar-bg: #1c1c1c;
            --sidebar-border: #444;
            --footer-bg: #1e1e1e;
            --footer-text: #ffffff;
        }
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
        background-color: var(--body-bg) !important;
        color: var(--body-text) !important;
    }

    /* Header - Tetap di atas dan full width */
    header[data-testid="stHeader"] {
        background-color: var(--header-bg) !important;
        color: var(--header-text) !important;
        text-align: center !important;
        font-size: 24px !important;
        font-weight: bold !important;
        padding: 20px 10px !important;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    header[data-testid="stHeader"]::before {
        content: "CORROSION IMAGE SEGMENTATION";
        display: block;
        font-size: 12px;
        font-weight: bold;
        color: var(--header-text);
        text-align: center;
        white-space: normal;
        word-wrap: break-word;
        padding: 4px;
    }

    /* Responsive Header Font */
    @media screen and (max-width: 768px) {
        header[data-testid="stHeader"]::before {
            font-size: 16px;
        }

        header[data-testid="stHeader"] {
            padding: 24px 12px !important;
        }
    }

    /* Main body space */
    .main-content {
        padding-top: 80px;
        padding-bottom: 50px;
    }

    .main-content {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
    }

    @media (min-width: calc(54rem)) {
        .st-emotion-cache-1ibsh2c {
            padding-left: 10;
        }
    }

    .st-emotion-cache-1sb9xkt {
        width: 681.525px;
        position: relative;
        flex: 1 1 0%;
        flex-direction: column;
        gap: 1rem;
    }

    .st-emotion-cache-1yiq2ps {
        display: flex;
        flex-direction: row;
        place-content: flex-start;
        align-items: stretch;
        position: absolute;
        inset: 0px;
        overflow: hidden;
        background-color: var(--body-bg) !important;
        color: var(--body-text) !important;
    }

    .st-emotion-cache-kgpedg {
        padding: 8px 16px 12px;
        background-color: var(--header-bg);
        color: var(--header-text);
        display: flex;
        align-items: center;
        flex-direction: row-reverse;
        font-weight: bold;
        font-size: 16px;
    }

    .st-emotion-cache-kgpedg::after {
        content: "Deep Learning - Corrosion Image Segmentation";
        margin-right: auto;
        line-height: 1.2;               /* Ini bikin jarak antar baris lebih rapat */
        white-space: normal;            /* Pastikan teks wrap alami, bukan satu baris */
        max-width: 80%;                 /* Biar ga terlalu panjang ke kanan */
    }


    /* Sidebar */
    [data-testid="stSidebar"] {
        max-width: 270px !important;
        background-color: var(--sidebar-bg) !important;
        color: var(--body-text) !important;
        line-height: 3;
        border-right: 6px solid var(--sidebar-border) !important;
    }

    /* Responsive sidebar */
    @media screen and (max-width: 768px) {
        .sidebar .sidebar-content {
            width: 100% !important;
        }
    }

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: var(--footer-bg);
        color: var(--footer-text);
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
    }
    </style>
    <div class="footer">Made by David Mario Yohanes Samosir | UNDIKSHA | 2025</div>
    """,
    unsafe_allow_html=True
)

# Fungsi Menu Segmentation
## Fungsi Tampilan Info Gambar
def display_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Membaca gambar
        st.text("Image Processing...")
        progress = st.progress(0)
        
        # Simulasi pemrosesan
        time.sleep(2)  # Misalnya proses selama 2 detik
        progress.progress(100)
        
        image = load_image(uploaded_file)
        
        # Membuat container dengan dua kolom
        with st.container():
            col1, col2 = st.columns(2)

            # Kolom pertama untuk visualisasi gambar
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            # Kolom kedua untuk informasi gambar
            with col2:
                # Mengambil informasi gambar
                resolution = f"{image.width} x {image.height}"
                color_mode = image.mode
                image_format = image.format

                # Membuat card informasi
                st.markdown(
                    f"""
                    <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                        <h4>Image Details</h4>
                        <ul>
                            <li><b>Resolution:</b> {resolution}</li>
                            <li><b>Color Mode:</b> {color_mode}</li>
                            <li><b>Image Format:</b> {image_format}</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.write("Please upload an image first!")
    return

# Fungsi untuk memuat model TensorFlow Keras
@st.cache_resource
def load_keras_model(model_path):
    """Fungsi untuk memuat model Keras dari path."""
    if not os.path.exists(model_path):
        st.error("Model file tidak ditemukan.")
        return None
    model = load_model(model_path)
    return model

## Fungsi Load Image
def load_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

## Colormap
### Define colormap for each class index
colormap = {
    0: [255, 0, 0],    # Red
    2: [0, 255, 0],    # Green
    1: [0, 0, 255],    # Blue
}

## Fungsi Preprocessing    
def preprocessing(uploaded_file):
    if uploaded_file is not None:
        # Menampilkan button untuk preprocessing
        if st.button("Preprocessing"):
            # Membaca file yang diunggah
            # Membaca gambar
            st.text("Image Processing...")
            progress = st.progress(0)
            
            # Simulasi pemrosesan
            time.sleep(2)  # Misalnya proses selama 2 detik
            progress.progress(100)
            
            image = load_image(uploaded_file)

            # Resize gambar ke ukuran 128x128
            resized_image = image.resize((256, 256))
            
            # Konversi gambar ke array uint8
            image_array = np.array(resized_image, dtype=np.uint8)
            
            # Menyimpan array ke st.session_state
            st.session_state.image_array = image_array
            
            # Menampilkan informasi array gambar
            st.write("Image Array Dimensions:", image_array.shape)  # Menampilkan dimensi
            st.write("Array Gambar (Sample Channel):", image_array[:, :, 0])  # Hanya channel R
            
            # Validasi apakah gambar berhasil dimuat
            if image_array is None:
                st.error("Error loading image.")
                return
            else:
                # Tampilkan gambar hasil unggahan
                st.image(image, caption=f"Image Resize {image_array.shape}", use_container_width=True)
                st.success("Image uploaded and processed successfully!")
            
            st.write("Gambar berhasil dimuat dalam format RGB.")

    else:
        st.info("Please upload an image file to proceed.")

    return

## Preprocessing and utility functions
def predict_image(image, model, colormap):
    """Predict image using the given model and apply specific class colors."""
    input_image = cv2.resize(image, (256, 256))  # Resize image to match model input size
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    prediction = model.predict(input_image)  # Get model prediction
    predicted_class = np.argmax(prediction[0], axis=-1)  # Get the class index with max probability

    # Create an RGB prediction map
    rgb_prediction = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_index, color in colormap.items():
        rgb_prediction[predicted_class == class_index] = color

    return rgb_prediction

## Fungsi Input Gambar
def show_segmentation():
    st.title("Upload and Display Image with Details")

    # Membuat pilihan input gambar (Upload Image atau kamera)
    input_choice = st.radio("Select the Image Input Method:", options=["Upload Image", "Take a Picture with Camera"])
    
    # Jika memilih "Upload Image"
    if input_choice == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="upload")

        if uploaded_file is not None:
            # Baca file sebagai bytes untuk keperluan perbandingan
            new_image_bytes = uploaded_file.read()

            # Cek apakah sudah pernah ada gambar yang diupload sebelumnya
            if "uploaded_image_bytes" not in st.session_state:
                # Jika belum, simpan gambar tersebut ke session_state
                st.session_state.uploaded_image_bytes = new_image_bytes
                # Pastikan state terkait proses juga direset
                st.session_state.image_array = None
                st.session_state.segmentation_result = None
                st.session_state.selected_model = None
            elif st.session_state.uploaded_image_bytes != new_image_bytes:
                # Jika gambar baru berbeda dengan yang lama, reset seluruh state yang dibutuhkan
                st.session_state.uploaded_image_bytes = new_image_bytes
                st.session_state.image_array = None
                st.session_state.segmentation_result = None
                st.session_state.selected_model = None
                # Jika ada variabel lain (misalnya flag proses) reset juga di sini
                st.session_state.processed = False
                st.session_state.tab1_selected = False
                st.session_state.tab2_selected = False


            # Karena st.file_uploader akan membaca file dan pointer sudah bergeser,
            # gunakan kembali BytesIO agar bisa dibaca lagi untuk proses selanjutnya.
            uploaded_file.seek(0)

        # Tampilkan gambar hasil unggahan
        display_uploaded_image(uploaded_file)
        preprocessing(uploaded_file)
        model()
    
    # Jika memilih "Take a Picture with Camera"
    elif input_choice == "Take a Picture with Camera":
        camera_image = st.camera_input("Take a Picture")

        display_uploaded_image(camera_image)
        preprocessing(camera_image)
        model()
    return

## overlay
def overlay_segmentation(original_image, segmentation, alpha=0.5):
    """Overlay segmentation onto the original image."""
    # Resize segmentation to match the original image size
    segmentation_resized = cv2.resize(segmentation, (original_image.shape[1], original_image.shape[0]))
    
    # Perform overlay
    return cv2.addWeighted(original_image, 1 - alpha, segmentation_resized, alpha, 0)

# Fungsi Model
def model():
    # Membuat 3 kolom untuk tampilan tab
    col1, col2 = st.columns(2)
    
    # Tombol dan status tab
    if "tab1_selected" not in st.session_state:
        st.session_state.tab1_selected = False
    if "tab2_selected" not in st.session_state:
        st.session_state.tab2_selected = False

    # Menambahkan styling untuk tombol menggunakan HTML dan CSS
    st.markdown("""
        <style>
            .stButton button {
                width: 100%;
                height: 60px;
                font-size: 18px;
                background-color.active:
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Menambahkan tombol atau teks di setiap kolom sebagai tab
    with col1:
        tab1 = st.button("🔴 Mobile U-Net")
        if tab1:
            st.session_state.tab1_selected = True
            st.session_state.tab2_selected = False
            st.session_state.tab3_selected = False

    with col2:
        tab2 = st.button("🟢 BiSeNetV3")
        if tab2:
            st.session_state.tab2_selected = True
            st.session_state.tab1_selected = False
            st.session_state.tab3_selected = False
            
    
    # Periksa apakah image_array tersedia
    if "image_array" not in st.session_state:
        st.warning("Please upload and preprocess an image first.")
        return
    
    # Ambil image_array dari session_state
    image_array = st.session_state.image_array

    # Konten berdasarkan tab yang dipilih
    if st.session_state.tab1_selected:
        st.write("Mobile U-Net with Accuracy **97,48%** and mIoU **92.41%** tab is selected.")
        mUnet = load_keras_model("./model/munet.keras")
        st.text("Segmentation Process...")
        time.sleep(2)

        # Catat waktu mulai
        start_time = time.time()

        # Segmentation using Mobile U-Net
        prediction = predict_image(image_array, mUnet, colormap)

        # Overlay hasil segmentasi
        overlay_image = overlay_segmentation(image_array, prediction)
        
        # Catat waktu selesai
        end_time = time.time()

        # Hitung waktu pemrosesan
        processing_time = end_time - start_time

        display_results(image_array, prediction, overlay_image, processing_time)

    # Konten berdasarkan tab yang dipilih
    if st.session_state.tab2_selected:
        st.write("BiSeNetV3 with Accuracy **98.70%** and mIoU **96.12%** tab is selected.")
        mUnet = load_keras_model("./model/bisenetv3.keras")
        st.text("Segmentation Process...")
        time.sleep(2)

        # Catat waktu mulai
        start_time = time.time()

        # Segmentation using Mobile U-Net
        prediction = predict_image(image_array, mUnet, colormap)

        # Overlay hasil segmentasi
        overlay_image = overlay_segmentation(image_array, prediction)
        
        # Catat waktu selesai
        end_time = time.time()

        # Hitung waktu pemrosesan
        processing_time = end_time - start_time

        display_results(image_array, prediction, overlay_image, processing_time)

## Display result
def display_results(image, prediction, overlay_image, processing_time):
    # Menampilkan hasil segmentasi di Streamlit
    st.write("### 📊 **Segmentation Results**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(prediction, caption="Segmented Image", use_container_width=True)
    with col3:
        st.image(overlay_image, caption="Overlay Image", use_container_width=True)

    # Tampilkan waktu proses
    st.write(f"Segmentation completed in **{processing_time:.2f} seconds**")
        
    st.success("Segmentation completed.")
    # Menyediakan tombol untuk mengunduh gambar hasil segmentasi
    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
    btn = st.download_button(
        label="Download Segmented Image",
        data=BytesIO(img_encoded.tobytes()),
        file_name="segmented_image.jpg",
        mime="image/jpeg"
    )

    # Menampilkan informasi piksel korosi
    st.write("Information Retrieval Process of Segmented Pixels...")
    time.sleep(2)
        
    st.write("### 📑 **Corrosion Pixel Information**")

    # Hitung jumlah piksel warna merah dan biru serta persentase korosi
    total_pixels = image.shape[0] * image.shape[1]
    red_pixel_count = np.sum((prediction == [255, 0, 0]).all(axis=-1))
    blue_pixel = np.sum((prediction == [0, 0, 255]).all(axis=-1))
    blue_pixel_count = red_pixel_count + blue_pixel
    corrosion_percentage = (red_pixel_count / blue_pixel_count) * 100 if blue_pixel_count > 0 else 0

    # Menampilkan tabel informasi piksel
    data = {
        "Keterangan": ["Total Number of Pixels", "Number of Blue Pixels (Pipe)", "Number of Red Pixels (Corrosion)", "Percentage of Corrosion on Pipe"],
        "Nilai": [total_pixels, blue_pixel_count, red_pixel_count, f"{corrosion_percentage:.2f}%"]
    }

    df = pd.DataFrame(data)
    st.table(df)

    st.success("Information created successfully.")

    # Menambahkan tombol download untuk file CSV
    # Mengonversi DataFrame ke format CSV
    csv = df.to_csv(index=False).encode('utf-8')

    # Tombol download
    st.download_button(
        label="📥 Download Corrosion Info as CSV",
        data=csv,
        file_name='corrosion_pixel_info.csv',
        mime='text/csv'
    )

# Color Palettes: https://colorhunt.co/palette/fef9e1e5d0aca31d1d6d2323

# --- 3️⃣ Tambahkan Sidebar dengan Navigasi ---
# Menambahkan sidebar untuk menu navigasi antara halaman "About" dan "Segmentation"
st.sidebar.title("📋 **MENU**")
st.sidebar.image("Segmentation.png", use_container_width=True)
menu = st.sidebar.radio("Choose Page:", ["About", "Segmentation"])

# --- 4️⃣ Tampilan Body Berdasarkan Menu ---
# Menampilkan konten utama berdasarkan pilihan menu
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Menu "About" - Menampilkan deskripsi proyek
if menu == "About":
    st.title("📖 Segmentation Project Information")
    st.write("---")
    
    # Membuat container untuk deskripsi proyek
    with st.container():
        st.subheader("💡 Project Description")
        st.write("""
        This project performs multi-class image segmentation to predict pipeline corrosion by comparing two deep learning model architectures, namely **Mobile U-Net + EfficientNetB1** and **BiSeNetV3 + EfficientNetB1**.
        The segmentation is performed with special coloring: `red` for corrosion area, `blue` for pipe, and `green` for background, and displays the segmentation result along with the information of corrosion percentage on the pipe visually.
        """)

    # Membuat dua kolom untuk tujuan dan teknologi yang digunakan
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔹 **Objective**")
        st.write("""
        - Implementing Deep Learning model on pipeline corrosion segmentation.
        - Comparing the efficiency and performance of the two models (Mobile U-Net and BiSeNetV3).
        - Calculating the accurate percentage of corrosion pixel area in pipe image.
        """)

    with col2:
        st.markdown("### 🔹 **Technology Used**")
        st.write("""
        - 🧠 **Deep Learning**
        - 🏗️ Streamlit untuk **UI**
        - 📦 Python Libraries **(Pandas, Numpy, OpenCV, TensorFlow)**
        - 🖼️ Dataset Collection **(Roboflow, Google Image, Photoshop)**
        """)

    # Menambahkan ekspander untuk informasi tambahan
    with st.expander("📌 Additional Details"):
        st.write("""
        The dataset used consists of images of pipes with varying degrees of corrosion, 
        and the segmentation model will distinguish corrosion areas from the rest of the pipe using a CNN approach 
        with the implementation of the EfficientNetB1 pre-train model.
        """)

# Menu "Segmentation" - Untuk upload gambar dan melakukan segmentasi
elif menu == "Segmentation":
    st.title("📷 Corrosion Segmentation in Images")

    show_segmentation()
    
# Menutup tag div untuk konten utama
st.markdown("</div>", unsafe_allow_html=True)
