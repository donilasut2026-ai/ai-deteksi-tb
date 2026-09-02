Python 3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
... from sklearn.metrics import accuracy_score
... import joblib
... 
... st.set_page_config(page_title="AI Deteksi TB Online", layout="wide")
... st.title("🫁 AI Deteksi TB Aktif vs Laten - Online")
... 
... tab1, tab2, tab3 = st.tabs(["1. Download Template", "2. Latih Model", "3. Prediksi Pasien"])
... 
... with tab1:
...     st.header("Download Template Excel")
...     template_data = {
...         'usia': [30, 45], 'bb_turun_kg': [0, 6], 'batuk_minggu': [0, 4],
...         'keringat_malam': [0, 1], 'demam_sore': [0, 1], 'kontak_tb': [0, 1],
...         'dm': [0, 1], 'led': [15, 60], 'nlr': [2.1, 5.5], 'hasil_rontgen_skor': [0, 3],
...         'label': [0, 2]
...     }
...     df_template = pd.DataFrame(template_data)
...     st.dataframe(df_template)
...     csv = df_template.to_csv(index=False).encode('utf-8')
...     st.download_button("📥 Download Template", csv, "template_tb.csv")
... 
... with tab2:
...     st.header("Upload Data dan Latih Model")
...     uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
...     if uploaded_file is not None:
...         df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
...         st.dataframe(df.head())
...         if st.button("🚀 Latih Model"):
...             X = df.drop('label', axis=1); y = df['label']
...             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
...             model = RandomForestClassifier(n_estimators=200, random_state=42)
...             model.fit(X_train, y_train)
...             acc = accuracy_score(y_test, model.predict(X_test))
...             st.success(f"Akurasi Model: {acc*100:.2f}%")
...             joblib.dump(model, 'model_tb_final.pkl')
... 
... with tab3:
...     st.header("Prediksi Pasien Baru")
...     try:
...         model = joblib.load('model_tb_final.pkl')
...         st.success("Model siap")
...     except:
...         st.warning("Latih model dulu di Tab 2")
...         model = None
...     if model:
...         # ... input form sama seperti sebelumnya ...
