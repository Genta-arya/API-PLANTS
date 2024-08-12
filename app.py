import os
import sys
import io
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import json

# Menonaktifkan logging TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hanya menampilkan error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load model
model = load_model('D:/Project/Joki Skripsi/Klasifikasi Penyakit Daun/API/mlearning/models/model_1.keras')

def predict_image(image_path, location):
    # Proses gambar untuk prediksi
    image = Image.open(image_path).convert("L").resize((28, 28))  # Mengubah gambar menjadi grayscale dan resize
    image_array = np.array(image) / 255.0  # Normalisasi gambar
    data_array = image_array.reshape(1, 28, 28, 1)  # Ubah bentuk menjadi (1, 28, 28, 1) untuk gambar grayscale

    # Melakukan prediksi
    predictions = model.predict(data_array)

    # Cek bentuk prediksi
    if predictions.size == 0:
        raise ValueError("Model did not return any predictions.")
    
    # Ambil kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)

    # Peta kelas ke nama penyakit
    class_names = ["late blight", "healthy", "early blight"]

    # Periksa apakah ada kelas yang diprediksi
    if predicted_class.size == 0 or predicted_class[0] >= len(class_names):
        raise ValueError("Hasil prediksi penyakit tidak ditemukan")

    label = class_names[predicted_class[0]]

    # Tambahkan informasi penyakit dan solusi
    disease_info = {
        "late blight": "Late blight adalah penyakit jamur yang menyerang daun, batang, dan buah kentang.",
        "healthy": "Tanaman kentang dalam keadaan sehat tanpa tanda-tanda penyakit.",
        "early blight": "Early blight adalah penyakit jamur yang menyebabkan bercak coklat pada daun."
    }
    solutions = {
        "late blight": "Gunakan fungisida berbasis tembaga setiap 7-10 hari sekali.",
        "healthy": "Tanaman dalam kondisi baik. Lanjutkan dengan rutinitas pemeliharaan standar.",
        "early blight": "Gunakan fungisida berbasis tembaga atau mankozeb setiap 7-10 hari sekali."
    }

    info = disease_info.get(label, "Informasi tidak tersedia.")
    solution = solutions.get(label, "Solusi tidak tersedia.")

    # Menambahkan teks kelas yang diprediksi ke gambar asli
    draw = ImageDraw.Draw(image)
    text = f"Predicted Class: {label}"
    draw.text((10, 10), text, fill="white") 

    results = {
        "predicted_class": label,
        "confidence": float(np.max(predictions)),
        "location": location,
        "info": info,
        "solution": solution
    }

    return json.dumps(results, ensure_ascii=False)  # Mengembalikan response JSON sebagai string

if __name__ == "__main__":
    image_path = sys.argv[1]
    location = sys.argv[2]

    try:
        result = predict_image(image_path, location)
        print(result)  # Hanya mencetak hasil JSON
    except ValueError as ve:
        print(json.dumps({"error": str(ve)}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
