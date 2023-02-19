from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import json

categories = ['Software', 'Hardware']
train = {
    "data": [
        "data muncul nomor periode penilaian pranota pegawai pelabuhan nama jpj tagihan permohonan project absen pkwt cuti pds surat pembayaran humanis penarikan approval gaji pkwtt ekontrak pengajuan email kontrak invoice shift ppn status",
        "printer internet laptop pc buka jaringan wifi cetak zoom lambat print connect lan office excel scan koneksi lokal kabel lemot garis word sinyal screen warna",
    ],
    "target": [0, 1]
}

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train["data"], train["target"])

def predict(text):
    labels = model.predict([text])
    return categories[labels[0]]
