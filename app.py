import sys
import os
import pandas as pd
import plotly.express as px

# Memaksa Python mengenali folder tempat file app.py ini berada
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import streamlit as st
from utils.preprocessing import bersihkan_teks
from utils.inference import deteksi_pelanggaran

# --- 1. Konfigurasi Halaman (Layout Wide) ---
st.set_page_config(page_title="Deteksi Pelanggaran Pemilu", page_icon="🗳️", layout="wide")

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; height: 50px; }
    .result-box-aman { padding: 20px; border-radius: 10px; background-color: #d4edda; color: #155724; border-left: 8px solid #28a745; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .result-box-bahaya { padding: 20px; border-radius: 10px; background-color: #f8d7da; color: #721c24; border-left: 8px solid #dc3545; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- 3. Sidebar ---
with st.sidebar:
    st.title("ℹ️ Tentang Sistem")
    st.info("Aplikasi *Text Mining* menggunakan arsitektur **IndoBERT** untuk mengklasifikasikan *tweet* terkait Pemilu 2024.")
    st.markdown("---")
    st.markdown("**Metode:**")
    st.markdown("- Preprocessing Teks\n- Model: IndoBERT Tweet\n- Klasifikasi: Potensi Pelanggaran vs Aman")

# --- 4. Header Utama ---
st.title("🚨 Dashboard Analisis Teks Pemilu 2024")
st.markdown("Mendeteksi potensi indikasi kecurangan atau pelanggaran pemilu pada media sosial.")
st.markdown("---")

# --- 5. Membuat Sistem Tab ---
tab1, tab2 = st.tabs(["🔍 Deteksi Teks Real-time", "📊 Hasil Riset Colab"])

# ==========================================
# TAB 1: DETEKSI REAL-TIME (Kode yang Lama)
# ==========================================
with tab1:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("### 📝 Input Teks Analisis")
        teks_input = st.text_area("Teks", label_visibility="collapsed", height=200, placeholder="Contoh: Ayo kawal kotak suara di TPS, jangan sampai ada kecurangan! @Bawaslu #Pemilu2024")
        submit_button = st.button("🔍 Mulai Analisis Teks", type="primary")

    with col2:
        st.markdown("### 📊 Panel Hasil Prediksi")
        if submit_button:
            if teks_input.strip() == "":
                st.warning("⚠️ Mohon ketikkan teks terlebih dahulu di kolom sebelah kiri.")
            else:
                with st.spinner("🧠 IndoBERT sedang bekerja menganalisis pola kalimat..."):
                    teks_bersih = bersihkan_teks(teks_input)
                    hasil, probabilitas = deteksi_pelanggaran(teks_bersih)
                    
                    st.markdown("#### Status Kalimat:")
                    if "Potensi" in hasil:
                        st.markdown(f'<div class="result-box-bahaya"><h3>⚠️ {hasil}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box-aman"><h3>✅ {hasil}</h3></div>', unsafe_allow_html=True)
                    
                    st.write("") 
                    st.metric(label="Tingkat Keyakinan Model (Confidence Score)", value=f"{probabilitas:.2f}%")
                    
                    with st.expander("🔎 Lihat Detail Preprocessing"):
                        st.write("**Teks Mentah:**", teks_input)
                        st.write("**Teks Bersih:**", teks_bersih)
        else:
            st.info("👈 Silakan masukkan teks dan klik tombol analisis.")


# ==========================================
# TAB 2: VISUALISASI HASIL RISET COLAB
# ==========================================
with tab2:
    st.markdown("### 📈 Ringkasan Hasil Dataset Riset")
    st.write("Visualisasi di bawah ini menampilkan distribusi data hasil klasifikasi yang dilakukan pada saat melatih model di Google Colab.")
    
    JUMLAH_PELANGGARAN = 156
    JUMLAH_AMAN = 72
    TOTAL_DATA = JUMLAH_PELANGGARAN + JUMLAH_AMAN
    
    # Membuat 3 kolom metrik angka
    m1, m2, m3 = st.columns(3)
    with m1:
        # BAGIAN YANG DIUBAH
        st.metric(label="Total Data Test", value=TOTAL_DATA)
    with m2:
        st.metric(label="Potensi Pelanggaran", value=JUMLAH_PELANGGARAN, delta="- Label 1", delta_color="inverse")
    with m3:
        st.metric(label="Aman / Netral", value=JUMLAH_AMAN, delta="+ Label 0", delta_color="normal")
        
    st.markdown("---")
    
    # Membuat Layout 2 kolom untuk Grafik
    chart_col1, chart_col2 = st.columns(2)
    
    # Siapkan Dataframe sederhana untuk grafik
    df_hasil = pd.DataFrame({
        "Kategori": ["Potensi Pelanggaran", "Aman (Bukan Pelanggaran)"],
        "Jumlah": [JUMLAH_PELANGGARAN, JUMLAH_AMAN]
    })
    
    with chart_col1:
        st.markdown("#### Distribusi Kelas (Pie Chart)")
        # Membuat Pie Chart interaktif dengan Plotly
        fig_pie = px.pie(
            df_hasil, 
            values='Jumlah', 
            names='Kategori', 
            color='Kategori',
            color_discrete_map={'Potensi Pelanggaran':'#ef553b', 'Aman (Bukan Pelanggaran)':'#00cc96'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### Perbandingan Jumlah (Bar Chart)")
        # Membuat Bar Chart interaktif dengan Plotly
        fig_bar = px.bar(
            df_hasil, 
            x='Kategori', 
            y='Jumlah', 
            color='Kategori',
            color_discrete_map={'Potensi Pelanggaran':'#ef553b', 'Aman (Bukan Pelanggaran)':'#00cc96'},
            text_auto=True
        )
        # Menghilangkan legend yang berulang di bar chart
        fig_bar.update_layout(showlegend=False) 
        st.plotly_chart(fig_bar, use_container_width=True)

    # (Opsional) Menampilkan Metrik Akurasi Model
    st.markdown("---")
    st.markdown("### 🎯 Performa Model IndoBERT")
    # Ubah angka ini sesuai dengan classification report (Akurasi, F1-Score) dari Colab-mu
    st.write("- **Akurasi Model:** 92.5%")
    st.write("- **F1-Score (Macro):** 91.8%")