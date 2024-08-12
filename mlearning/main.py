from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import uvicorn
from PIL import Image  # Untuk memproses gambar
from typing import List, Optional

app = FastAPI()

# Load model
model = load_model('D:/Project/Joki Skripsi/Klasifikasi Penyakit Daun/API/mlearning/models/model_1.keras')
class PredictionRequest(BaseModel):
    data: str
    location: Optional[str] = None

@app.post("/predict")
async def predict(file: UploadFile = File(...), data: str = "", location: Optional[str] = None):
    try:
        # Process data
        data_list = list(map(float, data.strip('[]').split(',')))
        data_array = np.array(data_list).reshape(1, -1)
        
        # Validate data shape
        if data_array.shape[1] != model.input_shape[1]:
            raise ValueError(f"Expected data shape: {model.input_shape[1]}, but got {data_array.shape[1]}")
        
        predictions = model.predict(data_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Process the image (Optional: if your model uses the image)
        image = Image.open(file.file)
        # Convert image to array or preprocess as required
        # image_array = np.array(image)

        # Map class to disease name
        class_names = ["late blight", "healthy", "early blight"]
        results = [
            {"class": class_names[idx], "confidence": round(100 * np.max(predictions[idx]), 2)}
            for idx, _ in enumerate(predicted_class)
        ]

        # Add disease info and solutions
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

        response = []
        for idx, result in enumerate(results):
            disease = result["class"]
            solution = solutions[disease]
            info = disease_info[disease]
            response.append({
                "class": disease,
                "confidence": result["confidence"],
                "solution": solution,
                "info": info
            })

        # Add location info if provided
        if location:
            response.append({"location": location, "message": "Lokasi diterima, pertimbangkan cuaca lokal dalam perawatan tanaman."})

        return {"results": response}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

