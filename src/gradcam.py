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

# Projenin kök dizinine ulaşana kadar üst klasörlere çıkma
BASE_DIRECTORY = current_file_path.parent.parent
DATA_DIRECTORY = BASE_DIRECTORY / "bonage_dataset"
MODELS_DIRECTORY = BASE_DIRECTORY / "src" / "models"
checkpoint_path = MODELS_DIRECTORY / "best_xception_multi_input.h5"

# Global sample değişkeni (Kaydetme isimlendirmesi için başlangıç değeri)
sample = {"id": "unknown"}


# --------------------------------------------------------------
# DENEY YARDIMCI FONKSİYONU: GÖRÜNTÜYÜ KAYDIRMA (SPATIAL SHIFT)
# --------------------------------------------------------------
def shift_image(img_path, shift_x, shift_y):
    """
    Görüntüyü verilen piksel miktarlarında sağa/sola ve yukarı/aşağı kaydırır.
    shift_x: Pozitif değerler sağa, negatif değerler sola kaydırır.
    shift_y: Pozitif değerler aşağı, negatif değerler yukarı kaydırır.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {img_path}")

    rows, cols, ch = img.shape

    # Afiş dönüşüm matrisi (Affine Transformation Matrix) M = [1, 0, tx], [0, 1, ty]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Kaydırma işlemini uygula (Boş kalan pikseller varsayılan olarak siyah olur)
    shifted_img = cv2.warpAffine(img, M, (cols, rows))

    # Orijinal klasörün içine geçici bir kopya olarak kaydet
    temp_path = Path(img_path).parent / f"temp_shifted_{Path(img_path).name}"
    cv2.imwrite(str(temp_path), shifted_img)

    return temp_path


# --------------------------------------------------------------
# GRAD-CAM ALGORİTMASININ TANIMLANMASI (GRAD-CAM HEATMAP)
# --------------------------------------------------------------
def make_gradcam_heatmap(img_array, gender_data, model, last_conv_layer_name):
    # Son konvolüsyon çıktıları ve model tahminlerini döndüren yardımcı model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Yapılacak işlemleri izlemek ve sonrasında gradyan hesabı için GradientTape başlatma
    with tf.GradientTape() as G_tape:
        last_conv_layer_output, preds = grad_model([img_array, gender_data], training=False)
        class_channel = preds[:, 0]

    # Tahminin son konvolüsyon katmanına göre gradyanlarını hesapla
    grads = G_tape.gradient(class_channel, last_conv_layer_output)

    # Her bir özellik haritasının gradyan ağırlığını (önem derecesini) hesaplama
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Özellik haritalarını kendi ağırlıklarıyla çarpıp birleştirme
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Negatif değerleri sıfırlama (ReLU) ve heatmap'i maksimum değere göre [0,1] aralığına ölçekleme
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy()


# --------------------------------------------------------------
# GÖRSELLEŞTİRME VE ÇIKTI ÜRETME (VISUALIZATION PIPELINE)
# --------------------------------------------------------------
def generate_and_display_gradcam(img_path, gender_val, actual_age, model_path):
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found! Please wait for training to complete: {model_path}")
        return None, None, None

    # Model ağırlıklarının yüklenmesi
    model = build_multi_input_model((128, 128))
    model.load_weights(model_path)

    last_conv_layer = "block13_sepconv2_act"  # Xception mimarisinin en son konvolüsyon katmanının adı

    # Resmi modele uygun boyuta yani 128x128'e getirme
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    gender_data = np.array([[gender_val]])
    predicted_age = model.predict([img_array, gender_data], verbose=0)[0][0]

    # Isı haritasını üretme
    heatmap = make_gradcam_heatmap(img_array, gender_data, model, last_conv_layer)

    # Orijinal resmi OpenCV ile oku
    orig_img = cv2.imread(str(img_path))
    orig_img = cv2.resize(orig_img, (500, 500))

    # Isı haritasını orijinal resim boyutuna büyüt
    heatmap_resized = cv2.resize(heatmap, (500, 500))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Isı haritasını renkli hale getir
    jet_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Orijinal resim ile renkli ısı haritasını karıştır
    superimposed_img = jet_heatmap * 0.4 + orig_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return orig_img, superimposed_img, predicted_age


# --------------------------------------------------------------
# GÖRSELLEŞTİRME VE PLOT ETME (MATPLOTLIB PLOTTING)
# --------------------------------------------------------------
def visualize_gradcam_results(orig_img, actual_age, predicted_age, superimposed_img, suffix=""):
    plt.figure(figsize=(12, 6))

    # Sol taraf: Saf Röntgen Görüntüsü
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original X-Ray {suffix}\nActual Bone Age: {actual_age} Months")
    plt.axis("off")

    # Sağ taraf: Modelin Odak Alanı
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM Attention Map {suffix}\nPredicted Bone Age: {predicted_age:.1f} Months")
    plt.axis("off")

    plt.tight_layout()

    # Görüntüyü otomatik kaydetme
    output_folder = BASE_DIRECTORY / "outputs" / "gradcam_results"
    output_folder.mkdir(parents=True, exist_ok=True)

    label = f"sample_{sample['id']}" if suffix == "" else f"sample_{sample['id']}_{suffix.lower().replace(' ', '_')}"
    output_plot_path = output_folder / f"gradcam_{label}.png"

    plt.savefig(output_plot_path, dpi=300)
    print(f"[SUCCESS] Grad-CAM visualization successfully saved to: {output_plot_path}")

    plt.show()


# --------------------------------------------------------------
# MODEL TAHMİNİ VE PIPELINE TETİKLEME (EXECUTION PIPELINE)
# --------------------------------------------------------------
def generate_and_visualize_gradcam(img_path, gender_val, actual_age, model_path, suffix=""):
    if not model_path.exists():
        print(f"[ERROR] Trained model file not found! Please wait for training to complete: {model_path}")
        return

    # Süreçleri tetikleme
    orig_img, superimposed_img, predicted_age = generate_and_display_gradcam(img_path, gender_val, actual_age,
                                                                             model_path)

    if orig_img is not None:
        # Sonuçları görselleştirme
        visualize_gradcam_results(orig_img, actual_age, predicted_age, superimposed_img, suffix=suffix)


# --------------------------------------------------------------
# TEST SELEKSİYONU VE TETİKLEME (MAIN EXECUTION)
# --------------------------------------------------------------
if __name__ == "__main__":
    test_split_path = DATA_DIRECTORY / "df_test_split.csv"

    if test_split_path.exists():
        df_test = pd.read_csv(test_split_path)
        sample = df_test.sample(n=1).iloc[0]

        print(f"[INFO] Random test sample selected. ID: {sample['id']}")

        # 1. ADIM: Orijinal Görüntü ile Grad-CAM Çalıştırılması
        print("\n--- Running Grad-CAM on Original Image ---")
        generate_and_visualize_gradcam(
            img_path=Path(sample['path']),
            gender_val=sample['gender_encoded'],
            actual_age=sample['boneage'],
            model_path=checkpoint_path,
            suffix="(Original)"
        )

        # 2. ADIM: Aynı Görüntüyü Sağa ve Aşağı Kaydırıp Test Etme (Spatial Shift)
        print("\n--- Running Spatial Shift Experiment ---")
        print("[INFO] Applying spatial shift (50px right, 30px down) to check coordinate bias...")

        try:
            # Resmi manipüle et ve geçici adresi al
            shifted_img_path = shift_image(sample['path'], shift_x=50, shift_y=30)

            # Kaydırılmış yeni görsel ile pipeline'ı tetikle
            generate_and_visualize_gradcam(
                img_path=shifted_img_path,
                gender_val=sample['gender_encoded'],
                actual_age=sample['boneage'],
                model_path=checkpoint_path,
                suffix="(Shifted)"
            )

            # Test bitince üretilen geçici görsel dosyasını sistemden temizle
            if shifted_img_path.exists():
                shifted_img_path.unlink()
                print("[INFO] Temporary shifted image cleaned up successfully.")

        except Exception as e:
            print(f"[ERROR] Failed during spatial shift test: {e}")

    else:
        print("[WARNING] 'df_test_split.csv' not found. Please wait for main.py to create this file.")