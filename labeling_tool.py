import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_ciede2000

# Fungsi dari skrip training, kita masukkan ke sini agar mandiri
def extract_color_features_from_image(image_bgr):
    try:
        if image_bgr is None: return None
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
        
        height, width, _ = dipstick_crop.shape
        panel_width = width // 10
        if panel_width == 0: return None

        color_features = []
        for i in range(10):
            startX = i * panel_width
            endX = (i + 1) * panel_width
            roi = dipstick_crop[int(height*0.2):int(height*0.8), startX:endX]
            if roi.size == 0: continue
            avg_color = cv2.mean(roi)[:3] # Ambil BGR
            color_features.append(avg_color)
        
        return color_features if len(color_features) == 10 else None
    except Exception:
        return None

# Fungsi untuk menghitung jarak warna
def calculate_color_distance(color1_bgr, color2_bgr):
    # Konversi BGR ke LAB color space untuk perbandingan akurat
    color1_lab = rgb2lab(cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2RGB))
    color2_lab = rgb2lab(cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2RGB))
    return deltaE_ciede2000(color1_lab, color2_lab)[0][0]

# --- Definisi ROI dari gambar acuan (TETAP SAMA) ---
acuan_roi = {
    "Leukocytes": [("Neg.", (200, 48, 50, 25)), ("Trace", (300, 48, 50, 25)), ("Small", (375, 48, 50, 25)), ("Moderate", (445, 48, 50, 25)), ("Large", (520, 48, 50, 25))],
    "Nitrite": [("Neg.", (200, 118, 50, 25)), ("Positive", (335, 118, 235, 25))],
    "Urobilinogen": [("Normal", (230, 188, 50, 25)), ("16", (325, 188, 50, 25)), ("32", (400, 188, 50, 25)), ("64", (475, 188, 50, 25)), (">128", (550, 188, 50, 25))],
    "Protein": [("Neg.", (200, 253, 50, 25)), ("Trace", (300, 253, 50, 25)), ("+", (375, 253, 50, 25)), ("++", (445, 253, 50, 25)), ("+++", (520, 253, 50, 25))],
    "pH": [("5.0", (200, 318, 50, 25)), ("6.0", (260, 318, 50, 25)), ("6.5", (320, 318, 50, 25)), ("7.0", (380, 318, 50, 25)), ("7.5", (440, 318, 50, 25)), ("8.0", (500, 318, 50, 25)), ("8.5", (560, 318, 50, 25))],
    "Blood": [("Neg.", (200, 388, 80, 25)), ("Non hemolyzed", (288, 388, 80, 25)), ("Hemolyzed", (375, 388, 80, 25)), ("25 Small", (460, 388, 80, 25)), ("80 Moderate", (540, 388, 80, 25)), ("200 Large", (620, 388, 80, 25))],
    "Specific Gravity": [("1.000", (200, 453, 55, 25)), ("1.005", (265, 453, 55, 25)), ("1.010", (330, 453, 55, 25)), ("1.015", (395, 453, 55, 25)), ("1.020", (460, 453, 55, 25)), ("1.025", (525, 453, 55, 25)), ("1.030", (590, 453, 55, 25))],
    "Ketone": [("Neg.", (200, 520, 50, 25)), ("Trace", (300, 520, 50, 25)), ("Small", (375, 520, 50, 25)), ("Moderate", (445, 520, 50, 25)), ("Large", (520, 520, 50, 25))],
    "Bilirubin": [("Neg.", (200, 588, 50, 25)), ("Small", (375, 588, 50, 25)), ("Moderate", (445, 588, 50, 25)), ("Large", (520, 588, 50, 25))],
    "Glucose": [("Neg.", (200, 658, 50, 25)), ("Trace", (300, 658, 50, 25)), ("+", (375, 658, 50, 25)), ("++", (445, 658, 50, 25)), ("+++", (520, 658, 50, 25))]
}

def start_super_labeling_tool():
    # Muat gambar acuan dan ekstrak warna referensi
    acuan_img = cv2.imread("acuan_dipstick.jpg")
    if acuan_img is None:
        print("ERROR: File 'acuan_dipstick.jpg' tidak ditemukan.")
        return
    
    reference_colors = {}
    for param, choices in acuan_roi.items():
        reference_colors[param] = []
        for label_name, roi in choices:
            avg_color = cv2.mean(acuan_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])[:3]
            reference_colors[param].append({'label': label_name, 'color': avg_color})

    # Siapkan dataset dan file output
    dataset_path = "C:/Users/Rifqi/Downloads/Train_urin/dataset_augmented/"
    image_files = glob.glob(os.path.join(dataset_path, 'Foto*.jpg'))
    if not image_files:
        print(f"ERROR: Tidak ada gambar 'Foto*.jpg' ditemukan di folder '{dataset_path}'.")
        return

    output_filename = "labels.json"
    final_labels = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            final_labels = json.load(f)
        print(f"Memuat {len(final_labels)} label yang sudah ada dari {output_filename}.")

    plt.ion()
    fig, ax = plt.subplots()

    total_images = len(image_files)
    for i, img_path in enumerate(image_files):
        base_name = os.path.basename(img_path)
        if base_name in final_labels: continue

        print(f"\n{'='*50}\nMELABELI GAMBAR {i+1}/{total_images}: {base_name}\n{'='*50}")
        
        img_to_label = cv2.imread(img_path)
        # Ekstrak fitur warna dari gambar yang akan dilabeli
        detected_colors = extract_color_features_from_image(img_to_label)

        if not detected_colors:
            print(f"PERINGATAN: Gagal memproses gambar {base_name}. Melewati.")
            continue

        ax.clear()
        ax.imshow(cv2.cvtColor(img_to_label, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Gambar: {base_name}")
        plt.draw()
        plt.pause(0.1)

        current_image_labels = {}
        # Loop melalui 10 parameter
        for idx, (param, choices) in enumerate(reference_colors.items()):
            print(f"\n--- Parameter: {param} ---")
            
            # Dapatkan warna yang terdeteksi untuk parameter ini
            detected_color = detected_colors[idx]
            
            distances = []
            for ref_choice in choices:
                dist = calculate_color_distance(detected_color, ref_choice['color'])
                distances.append(dist)
            
            # Cari rekomendasi terbaik (jarak terendah)
            min_dist_idx = np.argmin(distances)
            recommendation = choices[min_dist_idx]['label']

            # Tampilkan pilihan dan rekomendasi
            prompt = []
            for choice_idx, ref_choice in enumerate(choices):
                marker = " <== REKOMENDASI" if choice_idx == min_dist_idx else ""
                prompt.append(f"  [{choice_idx}] {ref_choice['label']} (Jarak: {distances[choice_idx]:.2f}){marker}")
            
            print("\n".join(prompt))
            
            # Minta konfirmasi dari pengguna
            while True:
                user_input = input(f"Ketik nomor pilihan atau tekan ENTER untuk menerima rekomendasi [{min_dist_idx}]: ")
                if user_input == "":
                    chosen_idx = min_dist_idx
                    break
                try:
                    chosen_idx = int(user_input)
                    if 0 <= chosen_idx < len(choices):
                        break
                    else:
                        print("Pilihan tidak valid, coba lagi.")
                except ValueError:
                    print("Input harus berupa angka atau ENTER, coba lagi.")
            
            chosen_label = choices[chosen_idx]['label']
            current_image_labels[param] = chosen_label
            print(f"==> Anda memilih: {chosen_label}")

        final_labels[base_name] = current_image_labels
        with open(output_filename, 'w') as f:
            json.dump(final_labels, f, indent=4)
        print(f"\nProgres untuk {base_name} telah disimpan ke {output_filename}")

    plt.ioff()
    plt.close()
    print("\n\nPelabelan Selesai!")

if __name__ == '__main__':
    start_super_labeling_tool()