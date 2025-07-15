import json # ✨ FIX: Tambahkan import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import requests
from datetime import date, datetime, timedelta
from bs4 import BeautifulSoup

# === Inisialisasi Aplikasi ===
app = FastAPI(title="Smart Farmer - Plant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model AI Penyakit Tanaman ===
model = tf.keras.models.load_model("models/plant_disease_model_v2.keras")

# === Model AI Penyakit Tanaman ===
model = tf.keras.models.load_model("models/plant_disease_model_v2.keras")

# ✨ FIX 1: Logika ini sudah benar, tidak perlu diubah.
with open("models/labels.json") as f:
    class_indices = json.load(f)
    class_names = [""] * len(class_indices)
    for name, index in class_indices.items():
        class_names[index] = name

# ✨ FIX 2 (REFACTOR): Muat dataset baru ke dalam struktur yang lebih baik
disease_info_map = {}
with open("models/disease_dataset.json", encoding="utf-8") as f:
    disease_data = json.load(f)
    for item in disease_data:
        # Gunakan 'nama_penyakit' sebagai key
        disease_name = item['nama_penyakit']
        # Simpan semua informasi relevan
        disease_info_map[disease_name] = {
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

class WeeklyPlanRequest(BaseModel):
    nama_lahan: str
    fase: str
    modal: str
    strategi: str

registered_fields: List[FieldRegistration] = []

rencana_tanam = {
    "Padi": {
        1: "Persiapan lahan sawah & persemaian benih padi.",
        2: "Penanaman bibit padi (tandur) & pengairan awal.",
        3: "Penyiangan gulma & pemantauan serangan keong mas.",
        4: "Pemupukan pertama (Urea & SP-36).",
        5: "Pengendalian hama wereng & pemantauan air.",
        6: "Pemupukan kedua (Urea & KCL).",
        7: "Pemantauan penyakit blas & hawar daun.",
        8: "Pengisian bulir padi, jaga level air.",
        9: "Pemantauan hama burung & tikus.",
        10: "Mengeringkan sawah secara bertahap.",
        11: "Pemanenan padi.",
        12: "Penjemuran & pasca-panen.",
    },
    "Jagung": {
        1: "Pembajakan lahan kering & pembuatan bedengan.",
        2: "Penanaman biji jagung & pemupukan dasar (NPK).",
        3: "Penyiangan gulma & pembubunan tanah.",
        4: "Pengairan rutin & pemantauan ulat grayak.",
        5: "Pemupukan kedua (Urea).",
        6: "Pengendalian hama penggerek batang.",
        7: "Masa pembungaan, pastikan air cukup.",
        8: "Masa pengisian biji, waspada ulat tongkol.",
        9: "Pemantauan tingkat kematangan tongkol (masak susu).",
        10: "Pengeringan tanaman di lahan (jika perlu).",
        11: "Pemanenan tongkol jagung.",
        12: "Pengupasan & penjemuran biji jagung.",
    },
    "Cabai": {
        1: "Persemaian benih & sterilisasi media tanam.",
        2: "Persiapan lahan, pembuatan bedengan, & pemasangan mulsa.",
        3: "Pindah tanam bibit cabai ke lahan.",
        4: "Penyiraman rutin & pemasangan ajir (turus).",
        5: "Pemupukan pertama & perempelan tunas air.",
        6: "Pengendalian hama kutu kebul & trips.",
        7: "Pemupukan kedua & pemantauan penyakit antraknosa.",
        8: "Masa pembungaan & pembuahan awal.",
        9: "Panen pertama (petik hijau atau merah).",
        10: "Pemupukan setelah panen & pengendalian lalat buah.",
        11: "Panen rutin setiap 3-5 hari sekali.",
        12: "Pengelolaan pasca-panen & sortasi buah.",
    },
    "Default": {
        1: "Persiapan lahan dan benih.",
        2: "Penanaman dan penyiraman awal.",
        3: "Penyiangan gulma.",
        4: "Pemupukan dasar.",
        5: "Pengamatan hama dan penyakit.",
        6: "Penyemprotan pestisida bila perlu.",
        7: "Pemupukan lanjutan.",
        8: "Masa pertumbuhan vegetatif."
    }
}

# === Utilitas Cuaca (Tidak ada perubahan) ===
def get_time_period(hour):
    if 5 <= hour < 11: return "Pagi"
    elif 11 <= hour < 15: return "Siang"
    elif 15 <= hour <= 18: return "Sore"
    return None

def get_weather_advice(weather):
    weather = weather.lower()
    if "hujan" in weather:
        return "Waspadai hujan. Hindari penjemuran dan penyemprotan."
    elif "cerah" in weather:
        return "Cocok untuk menyemprot, panen, dan pengeringan hasil panen."
    elif "berawan" in weather:
        return "Masih aman untuk aktivitas ringan. Perhatikan perubahan cuaca."
    return "Periksa langsung kondisi cuaca di lapangan."

def fetch_weather(lat, lon, days=1):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,weathercode"
            f"&forecast_days={days}&timezone=Asia/Jakarta"
        )
        data = requests.get(url).json()
        result = {}

        for t, temp, code in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"], data["hourly"]["weathercode"]):
            dt = datetime.fromisoformat(t)
            hour = dt.hour
            period = get_time_period(hour)

            if dt.date() == date.today():
                key = "hari_ini"
            elif dt.date() == date.today() + timedelta(days=1):
                key = "besok"
            else:
                continue

            if key not in result:
                result[key] = {}
            
            weather_codes = {
                0: "Cerah",
                1: "Berawan", 2: "Berawan", 3: "Berawan",
                61: "Hujan", 63: "Hujan", 65: "Hujan",
                80: "Hujan", 81: "Hujan", 82: "Hujan"
            }
            
            if period and period not in result[key]:
                cuaca = weather_codes.get(code, "Cuaca Tidak Diketahui")
                priority = "danger" if "hujan" in cuaca.lower() else "warning" if "berawan" in cuaca.lower() else "good"
                
                result[key][period] = {
                    "cuaca": cuaca,
                    "suhu": f"{round(temp)}°C",
                    "nasihat": get_weather_advice(cuaca),
                    "priority": priority
                }
        return result
    except Exception:
        return {}

def reverse_geocode(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        headers = {"User-Agent": "SmartFarmerApp"}
        response = requests.get(url, headers=headers)
        data = response.json()
        kota = data.get("address", {}).get("county") or data.get("address", {}).get("state")
        return kota if kota else "Tidak diketahui"
    except Exception:
        return "Tidak diketahui"

# === Endpoint AI (Tidak ada perubahan di dalam fungsi) ===
# === Endpoint AI ===
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        img = np.expand_dims(np.array(image) / 255.0, 0)
        pred = model.predict(img)[0]
        
        label = class_names[np.argmax(pred)]
        confidence = round(float(np.max(pred)) * 100, 2)
        
        # ✨ DIUBAH: Ambil informasi lengkap dari map yang baru
        info = disease_info_map.get(label, {
            "deskripsi": "Informasi detail untuk penyakit ini belum tersedia.",
            "obat_rekomendasi": []
        })
        
        # ✨ DIUBAH: Kembalikan respons yang lebih terstruktur
        return {
            "label": label,
            "confidence": confidence,
            "deskripsi": info.get("deskripsi"),
            "obat_rekomendasi": info.get("obat_rekomendasi")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Endpoint Lainnya (Tidak ada perubahan) ===
@app.get("/fields")
def list_fields():
    return registered_fields

@app.post("/register-field")
def register_field(data: FieldRegistration):
    registered_fields.append(data)
    return {"status": "sukses", "pesan": "Lahan berhasil didaftarkan.", "data": data}

# === Endpoint Rencana Tani ===
@app.get("/weekly-plan-summary")
def weekly_plan_summary(nama_lahan: str):
    field = next((f for f in registered_fields if f.nama_lahan.lower() == nama_lahan.lower()), None)
    if not field:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan.")

    # ✨ DIUBAH: Logika untuk mendapatkan rencana yang spesifik
    # 1. Dapatkan rencana untuk tanaman yang dipilih, atau gunakan rencana "Default"
    plan_for_crop = rencana_tanam.get(field.tanaman, rencana_tanam["Default"])

    hari_tanam = (date.today() - field.tanggal_tanam).days
    summary = []
    for i in range(7):
        day_date = date.today() + timedelta(days=i)
        nama_hari = day_date.strftime("%A").capitalize()
        minggu_ke = max(1, ((hari_tanam + i) // 7) + 1)

        # 2. Ambil tugas dari rencana yang sudah spesifik berdasarkan minggu ke-
        tugas = plan_for_crop.get(minggu_ke, "Aktivitas bebas / pemeliharaan rutin.")
        summary.append({"hari": nama_hari, "nama_tugas": tugas})

    return summary

# Endpoint keseluruhan rencana tanu
@app.get("/full-plan/{nama_lahan}")
def get_full_plan(nama_lahan: str):
    """Mengembalikan seluruh rencana tanam (per minggu) untuk lahan spesifik."""
    field = next((f for f in registered_fields if f.nama_lahan.lower() == nama_lahan.lower()), None)
    if not field:
        raise HTTPException(status_code=404, detail="Lahan tidak ditemukan.")
    
    # Dapatkan rencana spesifik untuk tanaman, atau gunakan rencana "Default"
    plan_for_crop = rencana_tanam.get(field.tanaman, rencana_tanam["Default"])
    
    # Kembalikan seluruh objek rencana untuk tanaman tersebut
    return {
        "nama_tanaman": field.tanaman,
        "rencana": plan_for_crop
    }

# === Dashboard Harian ===
@app.get("/daily-dashboard")
def daily_dashboard(nama_lahan: str, lat: Optional[float] = Query(None), lon: Optional[float] = Query(None), kota: Optional[str] = Query(None)):
    field = next((f for f in registered_fields if f.nama_lahan.lower() == nama_lahan.lower()), None)
    if not field:
        raise HTTPException(status_code=404, detail=f"Lahan '{nama_lahan}' tidak ditemukan.")

    if lat is None or lon is None:
         # Logika geocoding untuk kota
        if not kota:
            raise HTTPException(status_code=400, detail="Parameter lokasi (lat/lon atau kota) harus disediakan.")
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={kota}"
            headers = {"User-Agent": "SmartFarmerApp/1.0"}
            geo_data = requests.get(geo_url, headers=headers).json()
            if not geo_data:
                raise HTTPException(status_code=404, detail=f"Koordinat kota '{kota}' tidak ditemukan.")
            lat, lon = float(geo_data[0]['lat']), float(geo_data[0]['lon'])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal mengambil data geolokasi: {e}")

    cuaca_data = fetch_weather(lat, lon, days=2)
    hari_tanam = (date.today() - field.tanggal_tanam).days
    minggu_ke = max(1, (hari_tanam // 7) + 1)

    # ✨ DIUBAH: Logika untuk mendapatkan tugas hari ini yang spesifik
    plan_for_crop = rencana_tanam.get(field.tanaman, rencana_tanam["Default"])
    tugas_hari_ini = plan_for_crop.get(minggu_ke, "Pemeliharaan rutin.")

    besok = cuaca_data.get("besok", {})
    peringatan = next(
        ({
            "pesan": "WASPADA HUJAN!",
            "saran": f"Diperkirakan hujan di waktu {period.lower()}. Lindungi alat dan tunda penyemprotan."
        } for period, info in besok.items() if info.get("priority") == "danger"),
        None
    )

    return {
        "sapaan": f"Hai Petani! Hari ke-{hari_tanam} untuk tanaman {field.tanaman} Anda.",
        "cuaca": cuaca_data.get("hari_ini", {}),
        "tugas_hari_ini": tugas_hari_ini,  # ✨ Menggunakan tugas yang sudah spesifik
        "last_update": datetime.now().strftime("%d %B %Y, %H:%M WIB"),
        "peringatan_besok": peringatan or {"pesan": "Cuaca besok normal.", "saran": "Tetap pantau kondisi cuaca."},
    }

@app.get("/weather-hourly")
def get_hourly_weather(lat: float = Query(...), lon: float = Query(...)):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            # ✨ TAMBAHKAN: precipitation_probability
            f"&hourly=weathercode,temperature_2m,precipitation_probability"
            f"&forecast_days=1&timezone=Asia/Jakarta"
        )
        
        data = requests.get(url).json().get("hourly", {})
        hourly_data = []

        # ✨ TAMBAHKAN: 'prob' untuk menampung probability
        for t, temp, code, prob in zip(
            data.get("time", []), 
            data.get("temperature_2m", []), 
            data.get("weathercode", []),
            data.get("precipitation_probability", [])
        ):
            dt = datetime.fromisoformat(t)
            
            # Ambil data hanya untuk 24 jam ke depan dari sekarang
            if dt < datetime.now(dt.tzinfo) or dt > datetime.now(dt.tzinfo) + timedelta(hours=24):
                continue
            
            # Interval per 3 jam
            if dt.hour % 3 == 0:
                weather_codes = {
                    0: "Cerah", 1: "Berawan", 2: "Berawan", 3: "Berawan",
                    61: "Hujan", 63: "Hujan", 65: "Hujan", 80: "Hujan", 81: "Hujan", 82: "Hujan"
                }
                cuaca = weather_codes.get(code, "Berawan")

                hourly_data.append({
                    "jam": dt.strftime("%H:%M"),
                    "cuaca": cuaca,
                    "suhu": round(temp), # ✨ Kirim sebagai angka, bukan string
                    "peluang_hujan": prob # ✨ TAMBAHKAN field baru ini
                })

        return {"status": "ok", "data": hourly_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Seeding Data Awal ===
# Di dalam main.py

@app.on_event("startup")
def seed_default_field():
    if not registered_fields:
        registered_fields.append(FieldRegistration(
            nama_lahan="Sawah Subur",
            tanaman="Padi",
            jenis_lahan="Sawah Irigasi",  # ✨ DITAMBAHKAN
            luas_m2=1000,
            tanggal_tanam=date(2025, 6, 1)
        ))
        registered_fields.append(FieldRegistration(
            nama_lahan="Kebun Jagung Manis",
            tanaman="Jagung",
            jenis_lahan="Tegal/Ladang", # ✨ DITAMBAHKAN
            luas_m2=500,
            tanggal_tanam=date(2025, 7, 1)
        ))
        registered_fields.append(FieldRegistration(
            nama_lahan="Lahan Cabai Merah",
            tanaman="Cabai",
            jenis_lahan="Perkebunan",    # ✨ DITAMBAHKAN
            luas_m2=200,
            tanggal_tanam=date(2025, 5, 20)
        ))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)