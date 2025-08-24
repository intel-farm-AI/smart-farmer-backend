# Smart Farmer - Plant Assistant API

Aplikasi backend berbasis FastAPI untuk deteksi penyakit tanaman, manajemen lahan, dan perencanaan aktivitas pertanian.

## Fitur Utama

- **Deteksi Penyakit Tanaman**: Upload gambar daun tanaman untuk prediksi penyakit menggunakan model AI.
- **Manajemen Lahan**: Registrasi dan daftar lahan pertanian.
- **Rencana Tanam Mingguan**: Dapatkan ringkasan aktivitas mingguan berdasarkan jenis tanaman dan tanggal tanam.
- **Dashboard Harian**: Informasi aktivitas harian, cuaca, dan rekomendasi untuk lahan tertentu.
- **Cuaca**: Prediksi cuaca harian dan per jam berdasarkan lokasi lahan.

## Struktur Direktori

```
main.py
Plant_Disease_Detection.ipynb
Readme.md
requirements.txt
models/
  model-v2/
    disease_dataset.json
    labels.json
    plant_disease_model_v2.keras
  model-v3/
    disease_label.json
    labels_id.json
    plant-disease-model-v3.bin
    plant-disease-model-v3.xml
```

## Instalasi

1. **Clone repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Jalankan aplikasi**
   ```sh
   uvicorn main:app --reload
   ```

## Endpoint Utama

- `POST /predict`  
  Upload gambar daun untuk prediksi penyakit.

## Model AI

Model deteksi penyakit tanaman disimpan di folder `models/model-v2/` dan dimuat otomatis saat aplikasi dijalankan.

## Catatan

- Pastikan file model dan dataset tersedia sesuai struktur di atas.
- API dapat diakses di `http://127.0.0.1:8000/docs` untuk dokumentasi interaktif Swagger.

---

Smart Farmer