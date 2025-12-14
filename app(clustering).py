# =====================================================
# HIERARCHICAL CLUSTERING – STREAMLIT FINAL
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
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
Aplikasi ini menampilkan **seluruh proses Hierarchical Clustering**  
mulai dari *data cleaning* hingga *evaluasi Silhouette Score*.
""")

# =====================================================
# LOAD DATA (FIX, TANPA UPLOAD)
# =====================================================
st.header("Load Dataset")

df = pd.read_csv("penjualan_tiket_pesawat.csv")

st.write("**Dataset Awal**")
st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================
st.header("Data Cleaning")

# Cek missing value
missing = df.isnull().sum()

st.write("**Jumlah Missing Value per Kolom**")
st.dataframe(missing)

# Hapus missing value (jika ada)
df_clean = df.dropna().reset_index(drop=True)

st.write("**Dataset Setelah Cleaning**")
st.dataframe(df_clean.head())

# =====================================================
# SELEKSI DATA NUMERIK
# =====================================================
st.header("Seleksi Fitur Numerik")

data = df_clean.select_dtypes(include=["int64", "float64"])

st.write("**Kolom Numerik yang Digunakan untuk Clustering:**")
st.write(list(data.columns))

# =====================================================
# NORMALISASI DATA
# =====================================================
st.header("Normalisasi Data (StandardScaler)")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

st.write("""
Normalisasi dilakukan agar semua fitur berada pada skala yang sama,
karena Hierarchical Clustering berbasis jarak.
""")

# =====================================================
# DENDROGRAM
# =====================================================
st.header("Dendrogram Hierarchical Clustering")

linked = linkage(scaled_data, method="ward")

fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(linked, ax=ax)
ax.set_title("Dendrogram Hierarchical Clustering")
ax.set_xlabel("Data")
ax.set_ylabel("Jarak")
st.pyplot(fig)

st.info("""
Dendrogram digunakan untuk melihat proses penggabungan data
dan membantu menentukan jumlah cluster yang optimal.
""")

# =====================================================
# EVALUASI SILHOUETTE (DINAMIS)
# =====================================================
st.header("Evaluasi Silhouette Score")

k = st.slider("Pilih Jumlah Cluster (k)", 2, 5, 3)

hc = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward",
    metric="euclidean"
)

labels = hc.fit_predict(scaled_data)

sil_score = silhouette_score(scaled_data, labels)

st.write(f"**Silhouette Score untuk k = {k}:** `{sil_score:.3f}`")

# =====================================================
# VISUALISASI CLUSTER
# =====================================================
st.header("Visualisasi Hasil Clustering")

df_clean["cluster"] = labels

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.scatterplot(
    x=df_clean[data.columns[0]],
    y=df_clean[data.columns[1]],
    hue=df_clean["cluster"],
    palette="Set2",
    ax=ax2
)
ax2.set_title("Visualisasi Cluster")
st.pyplot(fig2)

# =====================================================
# ANALISIS KARAKTERISTIK CLUSTER
# =====================================================
st.header("Analisis Karakteristik Cluster")

cluster_summary = df_clean.groupby("cluster")[data.columns].mean()

st.write("**Rata-rata Tiap Fitur pada Setiap Cluster**")
st.dataframe(cluster_summary)

# =====================================================
# SIMPAN HASIL
# =====================================================
st.header("Simpan Hasil Clustering")

output_df = df_clean.copy()

csv = output_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Hasil Clustering (CSV)",
    data=csv,
    file_name="hasil_hierarchical_clustering.csv",
    mime="text/csv"
)

# =====================================================
# KESIMPULAN
# =====================================================
st.header("Kesimpulan")

st.write("""
Metode **Hierarchical Clustering** berhasil mengelompokkan data penjualan tiket
berdasarkan kemiripan nilai numerik.

Evaluasi menggunakan **Silhouette Score** menunjukkan bahwa nilai k tertentu
memberikan kualitas cluster yang lebih baik.

Hasil clustering dapat digunakan untuk:
- Segmentasi penjualan
- Analisis perilaku transaksi
- Pengambilan keputusan bisnis
""")
