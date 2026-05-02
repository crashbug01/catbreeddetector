import cv2
import numpy as np
import pandas as pd

def extract_average_color(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or path is incorrect: {image_path}")

    # Menghitung rata-rata warna untuk setiap kanal (B, G, R)
    average_color_per_row = np.mean(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    average_color = average_color[::-1]  # Mengubah urutan dari BGR ke RGB

    # Mengembalikan rata-rata warna sebagai tuple (R, G, B)
    return tuple(average_color)

def extract_colors_from_images(image_count, image_prefix, image_suffix):
    # List untuk menyimpan hasil
    results = []

    # Loop melalui semua gambar
    for i in range(1, image_count + 1):
        image_path = f'{image_prefix}{i}{image_suffix}'
        try:
            average_color = extract_average_color(image_path)
            results.append((image_path, *average_color))
        except ValueError as e:
            print(e)

    # Membuat DataFrame dari hasil
    df = pd.DataFrame(results, columns=['Image Path', 'Average Red', 'Average Green', 'Average Blue'])
    return df

# Contoh penggunaan
image_count = 30  # Ubah sesuai dengan jumlah gambar
image_prefix = 'Dataset(ROI)\dataset'  # Prefix nama file gambar
image_suffix = '.png'  # Suffix nama file gambar (ekstensi)

# Mengekstrak average color dari gambar dan menampilkannya dalam bentuk tabel
df = extract_colors_from_images(image_count, image_prefix, image_suffix)
print(df)

# Menyimpan tabel ke file CSV (opsional)
df.to_csv('average_colors.csv', index=False)
