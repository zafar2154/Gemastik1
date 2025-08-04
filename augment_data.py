import cv2
import numpy as np
import os
import glob
import random

def augment_image(image):
    """
    Menerapkan serangkaian augmentasi acak pada sebuah gambar.
    """
    augmented_image = image.copy()

    # 1. Augmentasi Kecerahan dan Kontras
    # new_image = alpha * image + beta
    # alpha: Kontras (1.0 = normal), beta: Kecerahan (0 = normal)
    alpha = 1.0 + random.uniform(-0.1, 0.1) # Kontras bervariasi +/- 10%
    beta = random.randint(-15, 15)         # Kecerahan bervariasi +/- 15
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

    # 2. Augmentasi Rotasi
    height, width = augmented_image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(-3, 3) # Rotasi acak antara -3 dan +3 derajat
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    # 3. Augmentasi Pergeseran (Translation)
    x_shift = random.randint(-5, 5) # Geser horizontal +/- 5 piksel
    y_shift = random.randint(-5, 5) # Geser vertikal +/- 5 piksel
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    augmented_image = cv2.warpAffine(augmented_image, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

    return augmented_image

def main():
    input_folder = "dataset/"
    output_folder = "dataset_augmented/"
    # Jumlah variasi baru yang akan dibuat untuk SETIAP gambar asli
    num_augmentations_per_image = 4

    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' telah dibuat.")

    # Dapatkan semua path gambar dari folder input
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))
    if not image_paths:
        print(f"Tidak ada gambar .jpg ditemukan di '{input_folder}'.")
        return

    total_images = len(image_paths)
    print(f"Ditemukan {total_images} gambar. Memulai augmentasi...")

    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        
        print(f"  ({i+1}/{total_images}) Memproses: {base_name}")

        # Muat gambar asli
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"    -> Gagal memuat gambar. Melewati.")
            continue
            
        # 1. Salin gambar asli ke folder baru tanpa modifikasi
        cv2.imwrite(os.path.join(output_folder, base_name), original_image)

        # 2. Buat dan simpan versi augmentasi
        for j in range(num_augmentations_per_image):
            augmented = augment_image(original_image)
            new_filename = f"{name}_aug_{j+1}{ext}"
            cv2.imwrite(os.path.join(output_folder, new_filename), augmented)
    
    total_new_files = len(glob.glob(os.path.join(output_folder, '*.jpg')))
    print("\nProses augmentasi selesai!")
    print(f"Dataset baru berisi {total_new_files} gambar di folder '{output_folder}'.")

if __name__ == '__main__':
    main()