from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
#from fastapi.middleware.cors import CORSMiddleware
from flask_cors import CORS

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app)  # CORS'u aç

# Modeli yükle
model = tf.keras.models.load_model("best_model.h5")  # Modelin yolunu güncelle

# Sınıf isimleri
class_labels = ["Birinci Derece Yanık","İkinci Derece Derin Yanık", "İkinci Derece Yüzeysel Yanık", "Üçüncü Derece Yanık"]  

# Görseli ön işleme fonksiyonu
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Görseli aç ve RGB'ye çevir
    image = image.resize((640, 640))  # Modelin beklediği boyuta getir (224x224 örnek)
    image = np.array(image) / 255.0  # Normalizasyon
    image = np.expand_dims(image, axis=0)  # Modelin beklediği şekle getir (1, 224, 224, 3)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    print("Gelen İstek:", request.content_type)  # İstek türünü logla
    print("Form Verisi:", request.form)  # Form verisini logla
    print("Dosya Verisi:", request.files)  # Gelen dosyaları logla

    if "image" not in request.files:
        print("HATA: İstek içinde 'image' dosyası yok!")
        return jsonify({"error": "Görsel yüklenmedi"}), 400

    file = request.files["image"]
    print("Gelen dosya adı:", file.filename)  # Terminalde loglamak için
    #return jsonify({"message": "Dosya başarıyla alındı"})
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    # Modelden tahmin al
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions)
    confidence = float(predictions[0][class_index])

    return jsonify({"burn_depth": class_labels[class_index], "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Servisi başlat
