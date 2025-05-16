import os
import shutil
import psycopg2
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# PostgreSQL bağlantı bilgileri
DB_PARAMS = {
    'host': 'localhost',
    'dbname': 'BurnAnalysisDB',
    'user': 'postgres',
    'password': '',
    'port': 5432
}

# Yollar

BEST_MODEL_PATH = "best_model.h5"
MODEL_BACKUP_DIR = "old_versions"
NEW_IMAGES_DIR = "new_training_images"

def fetch_verified_untrained_data():
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT "PatientID", "PhotoPath", "BurnDepth" FROM "Patients" WHERE "Verified" = TRUE AND "Trained" = FALSE LIMIT 100')

            return cur.fetchall()

def mark_as_trained(patient_ids):
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.executemany('UPDATE "Patients" SET "Trained" = TRUE WHERE "PatientID" = %s', [(pid,) for pid in patient_ids])
        conn.commit()

UPLOADS_BASE_DIR = "C:/Users/zeyne/Downloads/burn-analysis-backend-main--1--main/burn-analysis-backend-main--1--main/burn-analysis-backend-main/BurnAnalysisBackend/wwwroot/uploads"  # Burayı senin sistemine göre değiştir.

def prepare_training_images(records):
    if os.path.exists(NEW_IMAGES_DIR):
        shutil.rmtree(NEW_IMAGES_DIR)
    os.makedirs(NEW_IMAGES_DIR)

    for pid, photo_path, label in records:
        absolute_photo_path = os.path.join(UPLOADS_BASE_DIR, os.path.basename(photo_path))
        if not os.path.exists(absolute_photo_path):
            print(f"⚠️ Uyarı: {absolute_photo_path} dosyası bulunamadı, atlanıyor.")
            continue
        dest_folder = os.path.join(NEW_IMAGES_DIR, label)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(absolute_photo_path, os.path.join(dest_folder, f"{pid}.jpg"))


def load_data():
    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
    train = datagen.flow_from_directory(NEW_IMAGES_DIR, target_size=(640, 640), class_mode='categorical', batch_size=8, subset='training')
    val = datagen.flow_from_directory(NEW_IMAGES_DIR, target_size=(640, 640), class_mode='categorical', batch_size=8, subset='validation')
    return train, val

def retrain_model(train, val):
    model = load_model(BEST_MODEL_PATH)
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = new_model.fit(train, validation_data=val, epochs=5)
    return new_model, history.history['val_accuracy'][-1]

def evaluate_model(model, val):
    return model.evaluate(val, verbose=0)[1]

def save_model_if_better(new_model, new_acc, old_acc):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    if new_acc > old_acc:
        backup_path = os.path.join(MODEL_BACKUP_DIR, f"model_backup_{timestamp}.h5")
        shutil.move(BEST_MODEL_PATH, backup_path)
        new_model.save(BEST_MODEL_PATH)
        print(f"✅ Yeni model kaydedildi. Eski model yedeklendi: {backup_path}")
        return True
    else:
        print("⚠️ Yeni model daha düşük doğrulukta. best_model.h5 korunuyor.")
        return False

def run_retraining_pipeline():
    print("🚀 Model yeniden eğitme süreci başlatılıyor...")
    data = fetch_verified_untrained_data()
    print(f"📦 Uygun yeni veri sayısı: {len(data)}")

    if len(data) < 15:
        print("🚫 Yeterli veri yok. En az 100 doğrulanmış, eğitilmemiş kayıt gerekir.")
        return

    prepare_training_images(data)
    train, val = load_data()

    new_model, new_acc = retrain_model(train, val)
    old_model = load_model(BEST_MODEL_PATH)
    old_acc = evaluate_model(old_model, val)

    print(f"🆕 Yeni model acc: {new_acc:.4f}, 🏅 Mevcut best_model acc: {old_acc:.4f}")
    is_better = save_model_if_better(new_model, new_acc, old_acc)

    if is_better:
        mark_as_trained([r[0] for r in data])
        print("📌 Veritabanı güncellendi: trained = TRUE")
    else:
        print("🚫 Veritabanı güncellenmedi. trained = FALSE olarak kalacak.")

if __name__ == "__main__":
    run_retraining_pipeline()

