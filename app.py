import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="AI Deteksi TB Online", layout="wide")
st.title("🫁 AI Deteksi TB Aktif vs Laten - Online")

tab1, tab2, tab3 = st.tabs(["1. Download Template", "2. Latih Model", "3. Prediksi Pasien"])

with tab1:
    st.header("Download Template Excel")
    template_data = {
        'usia': [30, 45], 'bb_turun_kg': [0, 6], 'batuk_minggu': [0, 4],
        'keringat_malam': [0, 1], 'demam_sore': [0, 1], 'kontak_tb': [0, 1],
        'dm': [0, 1], 'led': [15, 60], 'nlr': [2.1, 5.5], 'hasil_rontgen_skor': [0, 3],
        'label': [0, 2]
    }
    df_template = pd.DataFrame(template_data)
    st.dataframe(df_template)
    csv = df_template.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Template", csv, "template_tb.csv")

with tab2:
    st.header("Upload Data dan Latih Model")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        if st.button("🚀 Latih Model"):
            X = df.drop('label', axis=1); y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            st.success(f"Akurasi Model: {acc*100:.2f}%")
            joblib.dump(model, 'model_tb_final.pkl')

with tab3:
    st.header("Prediksi Pasien Baru")
    try:
        model = joblib.load('model_tb_final.pkl')
        st.success("✅ Model siap digunakan")
    except:
        st.warning("⚠️ Latih model dulu di Tab 2")
        model = None

    if model:
        st.markdown("#### Isi data pasien di sini")
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia", 18, 90, 30)
            bb_turun = st.number_input("Penurunan BB (kg)", 0.0, 20.0, 0.0, 0.1)
            batuk = st.number_input("Lama Batuk (minggu)", 0, 12, 0)
            keringat = st.selectbox("Keringat Malam", ["Tidak", "Ya"])
            demam = st.selectbox("Demam Sore", ["Tidak", "Ya"])

        with col2:
            kontak = st.selectbox("Riwayat Kontak TB", ["Tidak", "Ya"])
            dm = st.selectbox("Riwayat DM", ["Tidak", "Ya"])
            led = st.number_input("LED (mm/jam)", 5, 100, 20)
            nlr = st.number_input("NLR", 1.0, 10.0, 2.5, 0.1)
            rontgen = st.selectbox("Skor Rontgen", [0,1,2,3], format_func=lambda x: ["Normal", "Kecurigaan Ringan", "Kecurigaan Sedang", "Kecurigaan Tinggi"][x])

       if st.button("🔍 Prediksi Sekarang", type="primary"):
            # 1. Bikin dict dulu biar urutannya pasti sama
            input_dict = {
                'usia': usia, 
                'bb_turun_kg': bb_turun, 
                'batuk_minggu': batuk,
                'keringat_malam': 1 if keringat=="Ya" else 0,
                'demam_sore': 1 if demam=="Ya" else 0,
                'kontak_tb': 1 if kontak=="Ya" else 0,
                'dm': 1 if dm=="Ya" else 0,
                'led': led, 
                'nlr': nlr, 
                'hasil_rontgen_skor': rontgen
            }
            
            # 2. Ubah ke DataFrame dan paksa urutannya sama dengan waktu training
            feature_columns = ['usia','bb_turun_kg','batuk_minggu','keringat_malam','demam_sore','kontak_tb','dm','led','nlr','hasil_rontgen_skor']
            data_baru = pd.DataFrame([input_dict])[feature_columns]

            prediksi = model.predict(data_baru)[0]
            proba = model.predict_proba(data_baru)[0]
            label_map = {0: "Sehat", 1: "TB Laten", 2: "TB Aktif"}

            st.subheader("📋 Hasil Prediksi:")
            if prediksi == 2:
                st.error(f"**Hasil: {label_map[prediksi]}**")
                st.write(f"Probabilitas: {proba[prediksi]*100:.1f}%")
                st.write("**Rekomendasi:** Segera rujuk untuk tes TCM")
            elif prediksi == 1:
                st.warning(f"**Hasil: {label_map[prediksi]}**")
                st.write(f"Probabilitas: {proba[prediksi]*100:.1f}%")
                st.write("**Rekomendasi:** Pertimbangkan Terapi Pencegahan TB")
            else:
                st.success(f"**Hasil: {label_map[prediksi]}**")
                st.write(f"Probabilitas: {proba[prediksi]*100:.1f}%")
                st.write("**Rekomendasi:** Edukasi & pantau")
