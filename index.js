const express = require("express");
const { spawn } = require("child_process");
const multer = require("multer");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 8000;
const pathFolder = "E:\Project\React-Project\joki skripsi\MY Plant\API-PLANTS";
// Konfigurasi multer untuk mengupload file
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const tempDir = path.join(__dirname, "temp");
if (!fs.existsSync(tempDir)) {
  console.log(
    "Direktori temp tidak ada. Silakan buat direktori secara manual."
  );
}

app.post("/predict", upload.single("file"), (req, res) => {
  const location = req.body.location || "ketapang";
  const file = req.file;

  if (!file) {
    return res.status(400).json({ error: "File is required" });
  }

  // Simpan file gambar ke file sementara
  const tempFilePath = path.join(tempDir, file.originalname);
  fs.writeFileSync(tempFilePath, file.buffer);

  // Eksekusi skrip Python
  const pythonProcess = spawn("python", [
    // 'D:/Project/Joki Skripsi/Klasifikasi Penyakit Daun/API/app.py',
    path.join(__dirname, "app.py"),
    tempFilePath,
    location,
  ]);

  let result = "";
  pythonProcess.stdout.on("data", (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    // Hapus file sementara setelah eksekusi
    fs.unlinkSync(tempFilePath);

    if (code !== 0) {
      return res.status(500).json({ error: "Error executing Python script" });
    }

    // Menghapus karakter yang tidak diinginkan dari hasil
    result = result.replace(/\x1B\[[0-?9;]*[mG]/g, ""); // Hapus karakter kontrol ANSI
    result = result.trim(); // Menghapus spasi di awal dan akhir

    console.log("Raw Result from Python:", result); // Menampilkan hasil mentah dari Python

    // Mencari bagian JSON yang valid
    const jsonMatch = result.match(/(\{.*\})/);
    if (jsonMatch && jsonMatch[1]) {
      try {
        const jsonResult = JSON.parse(jsonMatch[1]);
        return res.json(jsonResult);
      } catch (e) {
        return res.status(500).json({
          error: "Error parsing JSON response",
          details: e.message,
          rawResult: result, // Menampilkan hasil mentah untuk debugging
        });
      }
    } else {
      return res.status(500).json({
        error: "No valid JSON found in the output",
        rawResult: result,
      });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
