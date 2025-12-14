# =====================================================
# REGRESI RANDOM FOREST – STREAMLIT FINAL
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Regresi Random Forest",
    layout="wide"
)

st.title("Regresi Random Forest – Penjualan Tiket Pesawat")
st.write("""
Aplikasi ini menampilkan **seluruh proses analisis regresi**
menggunakan **Random Forest Regressor**, mulai dari preprocessing
hingga evaluasi dan visualisasi hasil.
""")

# =====================================================
# LOAD DATA
# =====================================================
st.header("Load Dataset")

df = pd.read_csv("penjualan_tiket_pesawat.csv")

st.write("**Preview Dataset**")
st.dataframe(df.head())

# =====================================================
# CEK STRUKTUR DATA
# =====================================================
st.header("Struktur Dataset")

st.write("**Informasi Dataset**")
st.text(df.info(buf=None))

# =====================================================
# TARGET & FITUR
# =====================================================
st.header("Penentuan Target dan Fitur")

y = df["Total"]

X = df.drop(columns=[
    "Transaction_ID",
    "Date",
    "Total"
])

st.write("**Fitur yang Digunakan:**")
st.write(list(X.columns))

# =====================================================
# ENCODING DATA KATEGORIK
# =====================================================
st.header("Encoding Data Kategorikal")

encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = encoder.fit_transform(X[col])

st.write("Encoding selesai. Semua fitur kini bertipe numerik.")

# =====================================================
# SPLIT DATA
# =====================================================
st.header("Pembagian Data Train & Test")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

st.write(f"Data Train: {X_train.shape[0]} baris")
st.write(f"Data Test : {X_test.shape[0]} baris")

# =====================================================
# TRAINING MODEL
# =====================================================
st.header("Training Model Random Forest")

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.success("Model berhasil dilatih")

# =====================================================
# EVALUASI MODEL
# =====================================================
st.header("Evaluasi Model")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R²", f"{r2:.4f}")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.header("Feature Importance")

importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

# =====================================================
# VISUALISASI ACTUAL vs PREDICTED
# =====================================================
st.header("Visualisasi Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax1.set_xlabel("Nilai Aktual Total")
ax1.set_ylabel("Nilai Prediksi Total")
ax1.set_title("Actual vs Predicted Total Penjualan")
st.pyplot(fig1)

# =====================================================
# VISUALISASI FEATURE IMPORTANCE
# =====================================================
st.header("Grafik Feature Importance")

fig2, ax2 = plt.subplots(figsize=(8, 4))
importance_df.plot(
    x="Fitur",
    y="Importance",
    kind="bar",
    legend=False,
    ax=ax2
)
ax2.set_ylabel("Nilai Importance")
ax2.set_title("Feature Importance Random Forest Regressor")
st.pyplot(fig2)

# =====================================================
# DISTRIBUSI RESIDUAL
# =====================================================
st.header("Distribusi Error (Residual)")

residuals = y_test - y_pred

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.hist(residuals, bins=30)
ax3.set_xlabel("Residual (Error)")
ax3.set_ylabel("Frekuensi")
ax3.set_title("Distribusi Error Prediksi")
st.pyplot(fig3)

# =====================================================
# KESIMPULAN
# =====================================================
st.header("📌 Kesimpulan")

st.write("""
Berdasarkan hasil evaluasi model Random Forest Regressor, diperoleh nilai
Mean Absolute Error (MAE) sebesar 40.560 dan Root Mean Squared Error (RMSE)
sebesar 64.753. Nilai error tersebut relatif kecil dibandingkan skala nilai
total penjualan tiket, sehingga menunjukkan bahwa prediksi model memiliki
tingkat kesalahan yang rendah.

Selain itu, nilai koefisien determinasi (R²) sebesar 0.9993 menunjukkan bahwa
model mampu menjelaskan sekitar 99.93% variasi data total penjualan tiket
pesawat. Hal ini mengindikasikan bahwa hubungan antara fitur input dan variabel
target berhasil dipelajari dengan sangat baik oleh model.

Dengan demikian, model Random Forest Regressor dapat dikatakan memiliki
performa yang sangat baik dan andal dalam memprediksi total penjualan tiket
pesawat. Model ini layak digunakan sebagai alat bantu analisis dan pengambilan
keputusan, terutama dalam perencanaan dan evaluasi strategi penjualan.
""")
