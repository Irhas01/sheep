from rembg import remove
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

input_path = '16_crop.jpg' # input image path
output_path = 'output.png' # output image path

input = Image.open(input_path) # load image
output = remove(input) # remove background
output.save(output_path) # save image

import cv2
def estimate_weight(length_mm, breadth_mm):
    # Define regression equation parameters
    c = 0.064443
    d = 0.010059
    # c = 0.032
    # d = 0.01
    # Calculate the weight
    weight = c * breadth_mm + d * length_mm

    return weight
# Baca gambar
image = cv2.imread(output_path)
# image_cropped = image[155:853, 0:1280]
# Ubah gambar menjadi skala abu-abu
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Deteksi tepi menggunakan metode Canny
edges = cv2.Canny(gray, 0, 0)

# Operasi morfologi untuk mengisi celah
kernel = np.ones((10, 10), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Operasi morfologi untuk menghapus noise
kernel = np.ones((15, 15), np.uint8)
cleaned = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

segmented_frame = np.uint8(cleaned)
contours, _ = cv2.findContours(segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
            # Find the largest contour (assuming it's the sheep)
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the dimensions (length and breadth) of the sheep
            x, y, w, h = cv2.boundingRect(largest_contour)
            length_mm = w
            breadth_mm = h

            # Estimate weight
            estimated_weight = estimate_weight(length_mm, breadth_mm)
            
            # Display a message or draw a bounding box around the sheep
            cv2.putText(image, "Sheep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Weight: {estimated_weight:.2f} kg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Tampilkan gambar hasil
cv2.imshow("Tepi Gambar dengan Morfologi", image)

# Tunggu hingga pengguna menekan tombol apa pun
cv2.waitKey(0)

# Tutup semua jendela tampilan
cv2.destroyAllWindows()