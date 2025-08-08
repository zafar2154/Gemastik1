import tensorflow as tf
import os
import glob

# Dapatkan semua nama file .h5 di folder saat ini
h5_files = glob.glob('*.h5')

if not h5_files:
    print("Tidak ada file .h5 yang ditemukan di folder ini.")
else:
    print(f"Ditemukan {len(h5_files)} model .h5 untuk dikonversi.")

# Loop melalui setiap file .h5 dan konversi
for h5_file_path in h5_files:
    try:
        # Muat model Keras .h5
        model = tf.keras.models.load_model(h5_file_path)
        
        # Buat objek konverter TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Terapkan optimasi standar
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Lakukan konversi
        tflite_model = converter.convert()
        
        # Buat nama file output .tflite
        tflite_file_path = os.path.splitext(h5_file_path)[0] + '.tflite'
        
        # Simpan model .tflite
        with open(tflite_file_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Berhasil: '{h5_file_path}' -> '{tflite_file_path}'")
        
    except Exception as e:
        print(f"Gagal mengonversi {h5_file_path}: {e}")

print("\nProses konversi selesai.")