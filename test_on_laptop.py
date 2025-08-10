import os
import numpy as np
import cv2
import tensorflow as tf

# --- KONFIGURASI PENTING ---
# Arahkan ke model .h5 asli Anda
MODEL_FILES = {
    'Leukocytes': 'model_Leukocytes.h5',
    'Nitrite': 'model_Nitrite.h5',
    'Urobilinogen': 'model_Urobilinogen.h5',
    'Protein': 'model_Protein.h5',
    'pH': 'model_pH.h5',
    'Blood': 'model_Blood.h5',
    'Specific Gravity': 'model_Specific_Gravity.h5',
    'Ketone': 'model_Ketone.h5',
    'Bilirubin': 'model_Bilirubin.h5',
    'Glucose': 'model_Glucose.h5'
}

# ANDA HARUS MENGISI INI DENGAN BENAR SESUAI URUTAN KELAS SAAT TRAINING
# Ini harus sama persis dengan yang ada di main_app.py
LABEL_MAPS = {
    'Leukocytes': ['Neg.', 'Trace', 'Small', 'Moderate', 'Large'],
    'Nitrite': ['Neg.', 'Pos.'],
    'Urobilinogen': ['Normal', '16', '+', '++', '+++'],
    'Protein': ['Normal', 'Trace', '+', '++', '+++', '++++'],
    'pH': ['Normal', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5'],
    'Blood': ['Neg.', 'Non hemolyzed', 'Hemolyzed', '25 Small', '80 Moderate', '200 Large'],
    'Specific Gravity': ['Neg.', '1005', '1010', '1015', '1020', '1025', '1030'],
    'Ketone': ['Neg.', '0.5', '1.5', '4.0', '8.0', '16.0'],
    'Bilirubin': ['Neg.', 'Small', 'Moderate', 'Large'],
    'Glucose': ['Neg.', 'Trace', '+', '++', '+++', '++++']
}
# -----------------------------

# Salin fungsi 'extract_features' yang sama persis dari script 'final_training.py' Anda
def extract_features(image_bgr, advanced=True):
    # ... (Salin fungsi lengkapnya persis dari script training/main_app.py) ...
    # Pastikan fungsi ini mengembalikan list berisi 30 atau 60 fitur
    # FUNGSI INI HARUS ADA DI SINI
    try:
        if image_bgr is None: return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 20, 70)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        dipstick_crop = image_bgr[y:y+h, x:x+w]
        if dipstick_crop.size == 0: return None
        hsv_dipstick = cv2.cvtColor(dipstick_crop, cv2.COLOR_BGR2HSV)
        height, width, _ = hsv_dipstick.shape
        panel_width = width // 10
        if panel_width == 0: return None
        
        color_features = []
        for i in range(10):
            startX = i * panel_width
            endX = (i + 1) * panel_width
            roi = hsv_dipstick[int(height*0.2):int(height*0.8), startX:endX]
            if roi.size == 0: continue
            
            if advanced:
                mean, std_dev = cv2.meanStdDev(roi)
                features_per_panel = list(mean.flatten()) + list(std_dev.flatten())
                color_features.extend(features_per_panel)
            else:
                mean = cv2.mean(roi)
                color_features.extend(mean[:3])

        expected_length = 60 if advanced else 30
        return color_features if len(color_features) == expected_length else None
    except Exception:
        return None


# 1. Muat semua model Keras .h5
print("Memuat model Keras (.h5)...")
models = {}
for param, model_file in MODEL_FILES.items():
    if os.path.exists(model_file):
        # Gunakan tf.keras.models.load_model untuk file .h5
        models[param] = tf.keras.models.load_model(model_file)
        print(f"Model untuk '{param}' berhasil dimuat.")
    else:
        print(f"ERROR: File model '{model_file}' tidak ditemukan.")


def run_analysis_laptop(image_path):
    """Fungsi utama untuk menganalisis gambar di laptop."""
    print(f"\n Menganalisis gambar: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(" Gagal memuat gambar.")
        return None

    # 2. Ekstrak fitur dari gambar
    # Atur 'advanced' sesuai dengan cara Anda melatih model tersebut
    features = extract_features(image, advanced=False) # Ganti ke True jika Anda melatih dengan 60 fitur
    
    if features is None:
        print("Tidak dapat mengekstrak fitur dari gambar.")
        return None

    input_data = np.array([features], dtype=np.float32)
    final_results = {}

    # 3. Lakukan prediksi untuk setiap parameter
    for param, model in models.items():
        # Lakukan prediksi dengan model.predict()
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction[0])
        
        if param in LABEL_MAPS:
            predicted_label = LABEL_MAPS[param][predicted_index]
        else:
            predicted_label = f"Kelas Index {predicted_index} (Label tidak terdefinisi)"
        
        final_results[param] = predicted_label
    
    return final_results

# --- CONTOH PENGGUNAAN UTAMA ---
if __name__ == "__main__":
    # Ganti dengan path ke gambar yang ingin Anda uji di laptop
    # Pastikan gambar ini ada di folder yang benar
    test_image_path = 'fototest2.jpg' 

    if models:
        results = run_analysis_laptop(test_image_path)
        
        if results:
            print("\n--- HASIL ANALISIS (TES LAPTOP) ---")
            for parameter, value in results.items():
                print(f"- {parameter:<20}: {value}")
            print("------------------------------------")
    else:
        print("\nTidak ada model yang dimuat, program berhenti.")