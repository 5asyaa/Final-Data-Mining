# =====================================================
# HIERARCHICAL CLUSTERING – STREAMLIT FINAL (REVISED)
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Hierarchical Clustering Penjualan Tiket",
    layout="wide"
)

st.title("Hierarchical Clustering Penjualan Tiket Pesawat")
st.write("""
Aplikasi ini menampilkan proses **Hierarchical Clustering**
untuk segmentasi transaksi penjualan tiket pesawat
berdasarkan **jumlah tiket** dan **harga tiket**.
""")

# =====================================================
# LOAD DATA
# =====================================================
st.header("1️⃣ Load Dataset")

df = pd.read_csv("penjualan_tiket_pesawat.csv")

st.write("**Contoh Dataset**")
st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================
st.header("2️⃣ Data Cleaning")

missing = df.isnull().sum()
st.write("**Jumlah Missing Value per Kolom**")
st.dataframe(missing)

df_clean = df.dropna().reset_index(drop=True)

st.success("Dataset bersih dan siap digunakan.")

# =====================================================
# SELEKSI FITUR (PERBAIKAN METODOLOGI)
# =====================================================
st.header("3️⃣ Seleksi Fitur untuk Clustering")

st.write("""
Clustering hanya menggunakan **fitur independen**:
- `Ticket_Quantity`
- `Ticket_Price`

Fitur `Total` **tidak digunakan** karena merupakan hasil perkalian
kedua fitur tersebut dan dapat menyebabkan bias jarak.
""")

features = ["Ticket_Quantity", "Ticket_Price"]
data = df_clean[features]

st.write("**Fitur yang Digunakan:**")
st.write(features)

# =====================================================
# NORMALISASI DATA
# =====================================================
st.header("4️⃣ Normalisasi Data (StandardScaler)")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

st.write("""
Normalisasi dilakukan agar setiap fitur memiliki skala yang sama,
karena Hierarchical Clustering berbasis perhitungan jarak.
""")

# =====================================================
# DENDROGRAM
# =====================================================
st.header("5️⃣ Dendrogram Hierarchical Clustering")

linked = linkage(scaled_data, method="ward")

fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(linked, ax=ax)
ax.set_title("Dendrogram Hierarchical Clustering")
ax.set_xlabel("Data")
ax.set_ylabel("Jarak")
st.pyplot(fig)

st.info("""
Dendrogram digunakan untuk melihat struktur hierarki penggabungan data
dan memberikan gambaran awal jumlah cluster.
""")

# =====================================================
# EVALUASI SILHOUETTE SCORE
# =====================================================
st.header("6️⃣ Evaluasi Silhouette Score")

k = st.slider("Pilih Jumlah Cluster (k)", 2, 5, 4)

hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
labels = hc.fit_predict(scaled_data)

sil_score = silhouette_score(scaled_data, labels)

st.write(f"**Silhouette Score untuk k = {k}:** `{sil_score:.3f}`")

# =====================================================
# HASIL CLUSTERING
# =====================================================
st.header("7️⃣ Visualisasi Hasil Clustering")

df_clean["cluster"] = labels

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.scatterplot(
    data=df_clean,
    x="Ticket_Quantity",
    y="Ticket_Price",
    hue="cluster",
    palette="Set2",
    ax=ax2
)

ax2.set_title("Scatter Plot Hasil Hierarchical Clustering")
ax2.set_xlabel("Jumlah Tiket")
ax2.set_ylabel("Harga Tiket")
st.pyplot(fig2)

# =====================================================
# ANALISIS KARAKTERISTIK CLUSTER
# =====================================================
st.header("8️⃣ Analisis Karakteristik Tiap Cluster")

cluster_summary = (
    df_clean
    .groupby("cluster")[features]
    .mean()
    .rename(columns={
        "Ticket_Quantity": "Rata-rata Jumlah Tiket",
        "Ticket_Price": "Rata-rata Harga Tiket"
    })
)

st.write("**Rata-rata Nilai Tiap Cluster**")
st.dataframe(cluster_summary)

# =====================================================
# SIMPAN HASIL
# =====================================================
st.header("9️⃣ Simpan Hasil Clustering")

csv = df_clean.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Hasil Clustering (CSV)",
    data=csv,
    file_name="hasil_hierarchical_clustering.csv",
    mime="text/csv"
)

# =====================================================
# KESIMPULAN
# =====================================================
st.header("🔍 Kesimpulan")

st.write("""
Hierarchical Clustering berhasil mengelompokkan transaksi
penjualan tiket pesawat menjadi beberapa **segmen berdasarkan harga dan jumlah tiket**.

Hasil evaluasi Silhouette Score menunjukkan bahwa jumlah cluster yang dipilih
memberikan pemisahan cluster yang **cukup baik (moderat)**.

Segmentasi ini dapat dimanfaatkan untuk:
- Analisis pola pembelian
- Segmentasi transaksi
- Pengambilan keputusan bisnis berbasis nilai transaksi
""")
