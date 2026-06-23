# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
from src.models_architecture import build_multi_input_model
from pathlib import Path

# --------------------------------------------------------------
# DOSYA YOLLARI — ANA DİZİNE KİLİTLENME (ABSOLUTE BASE PATH)
# --------------------------------------------------------------
current_file_path = Path(__file__).resolve()

# Projenin  kök dizinine ulaşana kadar üst klasörlere çıkma
BASE_DIRECTORY = current_file_path.parent.parent
DATA_DIRECTORY = BASE_DIRECTORY / "bonage_dataset"
MODELS_DIRECTORY = BASE_DIRECTORY / "src"/ "models"
checkpoint_path = MODELS_DIRECTORY / "best_xception_multi_input.h5"

# --------------------------------------------------------------
# GRAD-CAM ALGORİTMASININ TANIMLANMASI (GRAD-CAM HEATMAP)
# --------------------------------------------------------------

# Isı haritası hesaplama
def make_gradcam_heatmap(img_array, gender_data, model, last_conv_layer_name):
    # Son konvolüsyon çıktıları ve model tahminlerini döndüren yardımcı model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Yapılacak işlemleri izlemek ve sonrasında gradyan hesabı için GradientTape başlatma
    with tf.GradientTape() as G_tape:
        last_conv_layer_output, preds = grad_model([img_array, gender_data])
        class_channel = preds[:, 0]

    # Tahminin son konvolüsyon katmanına göre gradyanlarını hesapla
    grads = G_tape.gradient(class_channel, last_conv_layer_output)

    # Her bir özellik haritasının gradyan ağırlığını (önem derecesini) hesaplama
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Özellik haritalarını kendi ağırlıklarıyla çarpıp birleştirme
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Negatif değerleri sıfırlama ve heatmap'i [0,1] aralığına normalize etme
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


# --------------------------------------------------------------
# GÖRSELLEŞTİRME VE ÇIKTI ÜRETME (VISUALIZATION PIPELINE)
# --------------------------------------------------------------
# Resmi OpenCV ile okuyup oluşturulan ısı haritasını röntgen görüntüsü üzerine yerleştirme
def generate_and_display_gradcam(img_path, gender_val, actual_age, model_path):
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found! Please wait for training to complete: {model_path}")
        return

        # Model ağırlıklarının yüklenmesi
    model = build_multi_input_model((128, 128))
    model.load_weights(model_path)

    last_conv_layer = "block14_sepconv2_act"  # Xception mimarisinin en son konvolüsyon katmanının adı

    # Resmi modele uygun boyuta yani 128x128'e getirme
    img = load_img(img_path, target_size=(128, 128))  # PIL Image nesnesinin okunması
    img_array = img_to_array(img)  # (128,128) -> (128,128,3)
    img_array = np.expand_dims(img_array, axis=0)  # Model tek bir görüntüyle bile çalışırken batch boyutu bekler
    img_array = preprocess_input(img_array)  # Ham pixel değerlerini Xception'ın beklediği aralığa [-1,1] dönüştürür.

    gender_data = np.array([[gender_val]])  # Cinsiyet verisi hazırlandı
    predicted_age = model.predict([img_array, gender_data])[0][0]  # Model tahmini

    # Isı haritasını üretme
    heatmap = make_gradcam_heatmap(img_array, gender_data, model, last_conv_layer)

    # resmi OpenCV ile okuyup oluşturulan ısı haritasını röntgen görüntüsü üzerine yerleştirme
    # Orijinal resmi OpenCV ile oku (Isı haritasını üzerine giydirmek için)
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, (500, 500))  # Boyut büyütüldü (128,128,3) -> (500,500,3)

    # Isı haritasını orijinal resim boyutuna büyüt
    heatmap_resized = cv2.resize(heatmap, (500, 500))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Isı haritasını renkli hale getir
    jet_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Orijinal resim ile renkli ısı haritasını %60 röntgen görüntüsü -%40 ısı haritası görüntüsü oranında karıştır
    superimposed_img = jet_heatmap * 0.4 + orig_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)  # [0,255] arası normalizasyon

    return orig_img, superimposed_img, predicted_age


# --------------------------------------------------------------
# GÖRSELLEŞTİRME VE PLOT ETME (MATPLOTLIB PLOTTING)
# --------------------------------------------------------------
def visualize_gradcam_results(orig_img, actual_age, predicted_age, superimposed_img):
    plt.figure(figsize=(12, 6))

    # Sol taraf: Saf Röntgen Görüntüsü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original X-Ray\nActual Bone Age: {actual_age} Months")
    plt.axis("off")

    # Sağ taraf: Modelin Odak Alanı
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Attention Map\nPredicted Bone Age: {predicted_age:.1f} Months")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# MODEL TAHMİNİ VE PIPELINE TETİKLEME (EXECUTION PIPELINE)
# --------------------------------------------------------------
def generate_and_visualize_gradcam(img_path, gender_val, actual_age, model_path):
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found! Please wait for training to complete: {model_path}")
        return

    # Süreçleri tetikleme
    orig_img, superimposed_img, predicted_age = generate_and_display_gradcam(img_path, gender_val, actual_age, model_path)

    # Sonuçları görselleştirme
    visualize_gradcam_results(orig_img, actual_age, predicted_age, superimposed_img)


# --------------------------------------------------------------
# TEST SELEKSİYONU VE TETİKLEME (MAIN EXECUTION)
# --------------------------------------------------------------
if __name__ == "__main__":
    test_split_path = DATA_DIRECTORY / "df_test_split.csv"

    if test_split_path.exists():
        df_test = pd.read_csv(test_split_path)
        sample = df_test.sample(n=1, random_state=42).iloc[0]

        print(f"[INFO] Random test sample selected. ID: {sample['id']}")
        generate_and_visualize_gradcam(
            img_path=sample['path'],
            gender_val=sample['gender_encoded'],
            actual_age=sample['boneage'],
            model_path=checkpoint_path
        )
    else:
        print("[WARNING] 'df_test_split.csv' not found. Please wait for main.py to create this file.")