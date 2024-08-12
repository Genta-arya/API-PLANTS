from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import uvicorn
from PIL import Image, ImageDraw, ImageFont  # Untuk memproses gambar dan menambahkan teks
import os

app = FastAPI()

# Load model
model = load_model('D:/Project/Joki Skripsi/Klasifikasi Penyakit Daun/API/mlearning/models/model_1.keras')

@app.post("/predict")
def predict(file: UploadFile = File(...), location: str = "ketapang"):
    try:
        # Proses gambar untuk prediksi
        original_image = Image.open(file.file)  # Gambar asli
        image = original_image.convert("L").resize((28, 28))  # Mengubah gambar menjadi grayscale dan resize
        image_array = np.array(image)  # Mengonversi gambar ke array numpy
        image_array = image_array / 255.0  # Normalisasi gambar
        data_array = image_array.reshape(1, 28, 28, 1)  # Ubah bentuk menjadi (1, 28, 28, 1) untuk gambar grayscale

        # Melakukan prediksi
        predictions = model.predict(data_array)

        # Print untuk memeriksa bentuk prediksi
        print("Predictions shape:", predictions.shape)
        print("Predictions values:", predictions)

        # Pastikan ada prediksi yang dihasilkan
        if predictions.size == 0:
            raise ValueError("Model did not return any predictions.")

        # Ambil kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(predictions, axis=1)

        # Print untuk memeriksa kelas yang diprediksi
        print("Predicted class index:", predicted_class)

        # Peta kelas ke nama penyakit
        class_names = ["late blight", "healthy", "early blight" , ]

        # Periksa apakah ada kelas yang diprediksi
        if predicted_class.size == 0 or predicted_class[0] >= len(class_names):
            raise ValueError("Deteksi tidak berhasil")

        label = class_names[predicted_class[0]]

        # Tambahkan informasi penyakit dan solusi
        disease_info = {
            "late blight": "Late blight adalah penyakit jamur yang menyerang daun, batang, dan buah kentang.",
            "healthy": "Tanaman kentang dalam keadaan sehat tanpa tanda-tanda penyakit.",
            "early blight": "Early blight adalah penyakit jamur yang menyebabkan bercak coklat pada daun, yang dapat meluas dan menyebabkan penurunan hasil panen."
        }
        solutions = {
            "late blight": "Gunakan fungisida berbasis tembaga setiap 7-10 hari sekali.",
            "healthy": "Tanaman dalam kondisi baik. Lanjutkan dengan rutinitas pemeliharaan standar.",
            "early blight": "Gunakan fungisida berbasis tembaga atau mankozeb setiap 7-10 hari sekali."
        }

        info = disease_info.get(label, "Informasi tidak tersedia.")
        solution = solutions.get(label, "Solusi tidak tersedia.")

        # Menambahkan teks kelas yang diprediksi ke gambar asli
        draw = ImageDraw.Draw(original_image)
  
        text = f"Predicted Class: {label}"
        draw.text((10, 10), text, fill="white") 

      
        results = [
            {"class": label, "confidence": round(100 * np.max(predictions[0]), 2), "solution": solution, "info": info}
        ]

        # Tambahkan informasi lokasi
        results.append({"location": location, "message": "Lokasi diterima, pertimbangkan cuaca lokal dalam perawatan tanaman."})

        return {"results": results}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    
    
    
    