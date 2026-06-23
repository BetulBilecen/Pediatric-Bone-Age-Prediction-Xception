# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path     # Dosya yollarını parents ile hiyerarşik şekilde yönetir.
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
from src.models_architecture import build_multi_input_model


# --------------------------------------------------------------
# DOSYA YOLLARI — ANA DİZİNE KİLİTLENME (ABSOLUTE BASE PATH)
# --------------------------------------------------------------

# .parent -> src klasörü | .parent.parent -> Projenin kök dizini (Pediatric-Bone-Age-Prediction-Xception)
current_file_path = Path(__file__).resolve()    # (src/evaluation.py)

# Projenin  kök dizinine ulaşana kadar üst klasörlere çıkma
BASE_DIRECTORY = current_file_path.parent.parent
DATA_DIRECTORY = BASE_DIRECTORY / "bonage_dataset"
MODELS_DIRECTORY = BASE_DIRECTORY / "src" / "models"
OUTPUTS_DIRECTORY = BASE_DIRECTORY / "outputs"

OUTPUTS_DIRECTORY.mkdir(parents=True, exist_ok=True)   # Çıktıların kaydedildiği klasör

checkpoint_path = MODELS_DIRECTORY / "best_xception_multi_input.h5"
test_split_path = DATA_DIRECTORY / "df_test_split.csv"

# --------------------------------------------------------------
# TEST SETİ DEĞERLENDİRME PIPELINE'I
# --------------------------------------------------------------
IMAGE_SIZE = (128, 128)

def run_evaluation():
    if not checkpoint_path.exists():
        print(f"[ERROR] Trained weights not found at: {checkpoint_path}")
        return None, None   # Veri ataması kırılması engellendi

    if not test_split_path.exists():
        print(f"[ERROR] Test split file not found at: {test_split_path}.\nRun main.py first.")
        return None, None   # Veri ataması kırılması engellendi

    print("[INFO] Loading test dataset metadata...")
    df_test = pd.read_csv(test_split_path)

    print("[INFO] Re-instantiating architecture and loading optimal weights...")

    model = build_multi_input_model(IMAGE_SIZE)
    model.load_weights(checkpoint_path)

    real_bone_ages = []
    predicted_ages = []

    print(f"[INFO] Generating predictions for {len(df_test)} test samples. Please wait...")

    # Tüm test verisindeki tahminleri alma
    for idx, row in df_test.iterrows():
        img_path = row["path"]
        gender_val = row["gender_encoded"]
        actual_bone_age = row["boneage"]

        # Görüntü yükleme ve Xception için ön işleme
        img = load_img(img_path,target_size= IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis= 0)
        img_array = preprocess_input(img_array)

        gender_data = np.array([[gender_val]])

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

    # X ve Y eksenlerindeki kemik yaşı ölçeklerini 25 aylık aralıklarla gösterme
    plt.xticks(np.arange(0, max_val + 25, 25))
    plt.yticks(np.arange(0, max_val + 25, 25))

    # Otomatik kaydetme adımı
    output_plot_path = OUTPUTS_DIRECTORY / "real_vs_pred_scatter.png"
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"[SUCCESS] Evaluation plot successfully saved to: {output_plot_path}")

    plt.show()
    return real_bone_ages, predicted_ages

# --------------------------------------------------------------
# BİAS TEŞHİS RAPORU (STATISTICAL BIAS DIAGNOSIS)
# --------------------------------------------------------------
if __name__ == "__main__":

    real_bone_ages, predicted_ages = run_evaluation()

    if real_bone_ages is not None:
        residual = predicted_ages - real_bone_ages
        df_analysis = pd.DataFrame({
            "Real": real_bone_ages,
            "Prediction": predicted_ages,
            "Residual": residual
        })

        print("\n" + "-" * 50)
        print("MATHEMATICAL BIAS DIAGNOSIS REPORT".center(50))
        print("-" * 50)

        # Tüm Test Seti Genelindeki Global Bias
        global_bias = np.mean(residual)
        print(f"Global Bias: {global_bias: .2f} months")

        if global_bias < 0:
            print("-> Status: Model has a global tendency to UNDERPREDICT !!")
        elif global_bias > 0:
            print("-> Status: Model has a global tendency to OVERPREDICT !!")
        else:
            print("-> Status: Model is approximately unbiased.")

        print("-" * 50)
        print("AGE-SEGMENTED BIAS BREAKDOWN:".center(50))
        print("-" * 50)

        # Yaş gruplarına göre Bias Kırılımı

        # Çocuklar (< 72 Ay)
        children_residuals = df_analysis[df_analysis['Real'] < 72]['Residual']  # Gerçek kemik yaşı 72 aydan küçük örneklerin residual (tahmin hatası) değerleri

        # Bu yaş grubunda örnek varsa bias değerini raporla
        if len(children_residuals) > 0:
            print(
                f" -> Children Bias (< 72 months)     : {np.mean(children_residuals):+.2f} months (n = {len(children_residuals)})")  # {...:+.2f} sonuç pozitifse ekrana otomatik olarak + işareti koyar

        # Orta yaş grubu (72-156 ay )
        mid_residuals = df_analysis[(df_analysis['Real'] >= 72) & (df_analysis['Real'] <= 156)]['Residual']
        if len(mid_residuals) > 0:
            print(f" -> Mid-Range (72-156 months) Bias  : {np.mean(mid_residuals):+.2f} months (n={len(mid_residuals)})")

        # Gençler ve Yaşlılar (>156 ay)
        adult_residuals = df_analysis[df_analysis['Real'] > 156]['Residual']
        if len(adult_residuals) > 0:
            print(f" -> Adolescents & Older (> 156 months) Bias       : {np.mean(adult_residuals):+.2f} months (n={len(adult_residuals)})")
