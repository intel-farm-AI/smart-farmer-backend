import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import numpy as np
import io
from datetime import date
from openvino import Core   # ✅ pakai import terbaru (bukan openvino.runtime)

# === Inisialisasi Aplikasi ===
app = FastAPI(title="Smart Farmer - Plant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model OpenVINO ===
ie = Core()
model = ie.read_model("models/model-v3/plant-disease-model-v3.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# === Labels ===
with open("models/model-v3/labels_id.json", encoding="utf-8") as f:
    class_indices = json.load(f)

# Buat list kosong dengan panjang sesuai jumlah kelas
class_names = [""] * len(class_indices)

for key, name in class_indices.items():
    # Ambil angka terakhir dari key (setelah underscore "_")
    # contoh: "PlantVillage_0" -> 0
    idx = int(key.split("_")[-1])
    class_names[idx] = name



# === Dataset Penyakit ===
disease_info_map = {}
with open("models/model-v3/disease_label.json", encoding="utf-8") as f:
    disease_data = json.load(f)
    for item in disease_data:
        disease_info_map[item['nama_penyakit']] = {
            "deskripsi": item['deskripsi'],
            "obat_rekomendasi": item['obat_rekomendasi']
        }

# === Model Data ===
class FieldRegistration(BaseModel):
    nama_lahan: str
    tanaman: str
    jenis_lahan: str
    luas_m2: int
    tanggal_tanam: date

registered_fields: List[FieldRegistration] = []

# === Endpoint Prediksi ===
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        img = np.expand_dims(np.array(image) / 255.0, 0).astype(np.float32)

        result = compiled_model([img])[output_layer]
        probs = np.squeeze(result)
        label_index = int(np.argmax(probs))
        confidence = round(float(np.max(probs)) * 100, 2)

        label = class_names[label_index]
        info = disease_info_map.get(label, {
            "deskripsi": "Informasi detail untuk penyakit ini belum tersedia.",
            "obat_rekomendasi": []
        })

        return {
            "label": label,
            "confidence": confidence,
            "deskripsi": info["deskripsi"],
            "obat_rekomendasi": info["obat_rekomendasi"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
