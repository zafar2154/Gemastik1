import json
import os
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===================================================================
# --- KONFIGURASI PELATIHAN ---
#
# PILIH PARAMETER YANG INGIN DILATIH
PARAMETER_TO_TRAIN = 'Glucose' 
#
# ATUR PENGGUNAAN FITUR:
# True  = Menggunakan 60 fitur (Rata-rata + Standar Deviasi)
# False = Hanya menggunakan 30 fitur (Rata-rata saja)
USE_ADVANCED_FEATURES = True
# ===================================================================

# Fungsi image processing yang sekarang lebih fleksibel
def extract_features(image_bgr, advanced=True):
    try:
        if image_bgr is None: return None
        # ... (Kode pre-processing: grayscale, blur, canny, crop tetap sama) ...
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 20, 70)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        padding = 5
        y_start, y_end = max(0, y - padding), min(image_bgr.shape[0], y + h + padding)
        x_start, x_end = max(0, x - padding), min(image_bgr.shape[1], x + w + padding)
        dipstick_crop = image_bgr[y_start:y_end, x_start:x_end]
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
                # Ambil rata-rata DAN standar deviasi
                mean, std_dev = cv2.meanStdDev(roi)
                features_per_panel = list(mean.flatten()) + list(std_dev.flatten())
                color_features.extend(features_per_panel)
            else:
                # Ambil rata-rata SAJA
                mean = cv2.mean(roi)
                color_features.extend(mean[:3]) # Hanya H, S, V

        # Pastikan jumlah fitur sesuai
        expected_length = 60 if advanced else 30
        return color_features if len(color_features) == expected_length else None
    except Exception:
        return None

# --- BAGIAN UTAMA SCRIPT ---

# 1. Muat Label
label_file = 'labels.json'
with open(label_file, 'r') as f:
    all_labels = json.load(f)

# 2. Siapkan Data
print(f"Mempersiapkan data untuk parameter: '{PARAMETER_TO_TRAIN}'")
print(f"Menggunakan Fitur Canggih: {USE_ADVANCED_FEATURES}")

X_data = []
y_labels_text = []
dataset_path = "dataset_augmented/"

for filename, labels in all_labels.items():
    img_path = os.path.join(dataset_path, filename)
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        # Panggil fungsi dengan flag dari konfigurasi
        features = extract_features(image, advanced=USE_ADVANCED_FEATURES)
        
        if features and PARAMETER_TO_TRAIN in labels:
            X_data.append(features)
            y_labels_text.append(labels[PARAMETER_TO_TRAIN])

if not X_data:
    print("ERROR: Tidak ada data yang berhasil disiapkan.")
    exit()

print(f"Total data yang siap dilatih: {len(X_data)} sampel.")
print("Distribusi Kelas Data:")
print(pd.Series(y_labels_text).value_counts())

# 3. Encoding & Pembagian Data
le = LabelEncoder()
y_data_encoded = le.fit_transform(y_labels_text)
y_data_categorical = to_categorical(y_data_encoded)
num_classes = len(le.classes_)
X_data = np.array(X_data)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data_categorical, test_size=0.20, random_state=42
)

# 4. Bangun Model (input_shape sekarang dinamis)
feature_count = 60 if USE_ADVANCED_FEATURES else 30
model = Sequential([
    Dense(256, activation='relu', input_shape=(feature_count,)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Latih Model
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.20,
    verbose=2,
    callbacks=[early_stopping]
)

# 6. Evaluasi dan Simpan
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi pada DATA UJI AKHIR: {accuracy * 100:.2f}%")
output_model_filename = f'model_{PARAMETER_TO_TRAIN}.h5'
model.save(output_model_filename)
print(f"Model telah disimpan sebagai '{output_model_filename}'")