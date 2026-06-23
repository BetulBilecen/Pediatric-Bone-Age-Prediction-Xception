# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
from src.models_architecture import build_multi_input_model


# --------------------------------------------------------------
# DOSYA YOLLARI — ANA DİZİNE KİLİTLENME (ABSOLUTE BASE PATH)
# --------------------------------------------------------------
current_file_path = os.path.abspath(__file__)

# Projenin  kök dizinine ulaşana kadar üst klasörlere çıkma
BASE_DIRECTION = current_file_path
while os.path.basename(BASE_DIRECTION) != "Pediatric-Bone-Age-Prediction-Xception":     BASE_DIRECTION = os.path.dirname(BASE_DIRECTION)
DATA_DIRECTION = os.path.join(BASE_DIRECTION, "bonage_dataset")
MODELS_DIRECTION = os.path.join(BASE_DIRECTION, "src", "models")
OUTPUTS_DIRECTION = os.path.join(BASE_DIRECTION, "outputs")

os.makedirs(OUTPUTS_DIRECTION, exist_ok=True)   # Çıktıların kaydedildiği klasör

checkpoin_path = os.path.join(MODELS_DIRECTION, "best_xception_multi_input.h5")
test_split_path = os.path.join(DATA_DIRECTION, "df_test_split.csv")

# --------------------------------------------------------------
# TEST SETİ DEĞERLENDİRME PIPELINE'I
# --------------------------------------------------------------
IMAGE_SIZE = (128, 128)

def run_evaluation():
    if not os.path.exists(checkpoin_path):
        print(f"[ERROR] Trained weights not found at: {checkpoin_path}")
        return

    if not os.path.exists(test_split_path):
        print(f"[ERROR] Test split file found at: {test_split_path}.\n Run main.py first.")
        return

    print("[INFO] Loading test dataset metadata...")
    df_test = pd.read_csv(test_split_path)

    print("[INFO] Re-instantiating architecture and loading optimal weights...")

    model = build_multi_input_model(IMAGE_SIZE)
    model.load_weights(checkpoin_path)

    real_bone_ages = []
    predicted_ages = []

    print(f"[INFO] Generating predictions for {len(df_test)} test samples. Please wait...")

    # Tüm test verisindeki tahminleri alma
    for idx, row in df_test.iterrows():
        img_path = row["path"]
        gender_val = row["gender_encoded"]
        actual_bone_age = row["boneage"]

        # Ön işlemi
        img = load_img(img_path,target_size= IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis= 0)
        img_array = preprocess_input(img_array)

        gender_data = np.array([gender_val])

        pred_age = model.predict([img_array, gender_data], verbose= 0)[0][0]

        # Listelere ekleme
        real_bone_ages.append(actual_bone_age)
        predicted_ages.append(pred_age)

    real_bone_ages = np.array(real_bone_ages)
    predicted_ages = np.array(predicted_ages)

# --------------------------------------------------------------
# SCATTER PLOT GÖRSELLEŞTİRME (MATPLOTLIB PLOTTING)
# --------------------------------------------------------------

    print("[INFO] Generating Real vs. Predicted scatter plot")
    plt.figure(figsize= (9,9))

    # Hasta noktalarını çizdirme
    plt.scatter(real_bone_ages, predicted_ages, alpha= 0.5, color= "royalblue", label = "Test Patients")

    # X = Y çizgisi
    max_val = max(max(real_bone_ages),max(predicted_ages))
    min_val = min(min(real_bone_ages),min(predicted_ages))
    plt.plot([min_val,max_val], [min_val, max_val], color= "crimson", linestyle= "--",linewidth= 2,label="Perfect Prediction (y = x)")

    # Grafik süslemeleri ve etiketleri
    plt.title("Pediatric Bone Age Regression Diagnosis\nReal vs. Predicted Performance Spectrum", fontsize=14, fontweight='bold')
    plt.xlabel("Real Bone Age (Months)", fontsize=12)
    plt.ylabel("Predicted Bone Age (Months)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)

    # Eksen sınırlandırması
    plt.xlim(min_val - 5, max_val + 5)
    plt.ylim(min_val - 5, max_val + 5)

    # Otomatik kaydetme adımı
    output_plot_path = os.path.join(OUTPUTS_DIRECTION, "real_vs_pred_scatter.png")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"[SUCCESS] Evaluation plot successfully saved to: {output_plot_path}")

    plt.show()

if __name__ == "__main__":
    run_evaluation()