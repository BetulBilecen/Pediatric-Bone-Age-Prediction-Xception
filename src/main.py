# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import os
from src.dataset import prepare_data
from src.models_architecture import build_multi_input_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# --------------------------------------------------------------
# SABİTLER VE HİPERPARAMETRELER (CONSTANTS & HYPERPARAMETERS)
# --------------------------------------------------------------

# Sabitler
IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 15
BASELINE_MAE_SCORE = 13.0 # Ay bazında baseline skoru

# Dosya yolları
CURRENT_DIRECTION = os.path.dirname(os.path.abspath(__file__))

# Ana proje klasörünü bulana kadar geriye git
BASE_DIR = CURRENT_DIRECTION
while os.path.basename(BASE_DIR) != "Pediatric-Bone-Age-Prediction-Xception":
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_DIRECTION = os.path.join(BASE_DIR, "bonage_dataset")
MODELS_DIRECTION = os.path.join(BASE_DIR, "src", "models")

# Klasör kontrolü
os.makedirs(MODELS_DIRECTION, exist_ok=True)
checkpoint_path = os.path.join(MODELS_DIRECTION, "best_boneage_model.keras")

# "models" klasörü yoksa oluştur
os.makedirs(MODELS_DIRECTION, exist_ok=True)
checkpoint_path = os.path.join(MODELS_DIRECTION, "best_xception_multi_input.h5")

# --------------------------------------------------------------
# VERİ JENERATÖRLERİNİN HAZIRLANMASI (DATA GENERATORS)
# --------------------------------------------------------------
print("[INFO] Loading datasets and preparing the multi-input data pipeline...")
train_gen, val_gen, num_train, num_val, df_test = prepare_data(DATA_DIRECTION, IMG_SIZE, BATCH_SIZE)

# ----------------------------------------------------------------------------
# MULTI-INPUT MODELİNİN İNŞA EDİLMESİ (MODEL ARCHITECTURE)
# ----------------------------------------------------------------------------
print("\n[INFO] Building the Multi-Input Architecture (Image + Gender) using the Functional API...")
model = build_multi_input_model(IMG_SIZE)
model.summary()

# ----------------------------------------------------------------------------
# CALLBACK MEKANİZMALARININ TANIMLANMASI (ROBUST CALLBACKS)
# ----------------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor= "val_loss",
        patience= 5,
        restore_best_weights= True
    ),

    ModelCheckpoint(
        filepath= checkpoint_path,
        monitor= "val_loss",
        save_best_only= True
    )
]

# ----------------------------------------------------------------------------
# MODEL EĞİTİMİ (MODEL TRAINING)
# ----------------------------------------------------------------------------
print("\n[INFO] Starting Multi-Input Model Training...")
history = model.fit(
    train_gen,
    steps_per_epoch = num_train // BATCH_SIZE,    # Bir epochta kaç batch işleneceğini belirler.
    validation_data = val_gen,
    validation_steps=num_val // BATCH_SIZE,  # Doğrulama sırasında kaç batch kullanılacağını belirtir.
    epochs = EPOCHS,
    callbacks = callbacks
)

# ----------------------------------------------------------------------------
# TEST VERİSİNİN SAKLANMASI (SAVING TEST SPLIT)
# ----------------------------------------------------------------------------
df_test.to_csv(os.path.join(DATA_DIRECTION, "df_test_split.csv"), index= False)
print(f"\n[SUCCESS] Training completed. Best model checkpoint saved to: {checkpoint_path}")

# ------------------------------------------------------------
# BASELINE KIYASLAMA RAPORU (BASELINE BENCHMARK REPORT)
# ------------------------------------------------------------

# Eğitim sırasında kaydedilen doğrulama MAE değerlerini alma
val_mae_history = history.history.get("val_mae")

print("\n" + "-" * 50)
print("BASELINE COMPARISON REPORT")
print("-" * 50)

# Val MAE metriğinin mevcut olup olmadığını kontrol etme
if val_mae_history is not None:

    # En düşük doğrulama MAE değerine sahip epoch'un indeksini bulma
    best_epoch_index = val_mae_history.index(
        min(val_mae_history)
    )

    # En iyi doğrulama MAE skorunu alma
    best_val_mae = val_mae_history[best_epoch_index]

    # Sonuçların raporlanması
    print(f"-> Current Baseline Val MAE : {BASELINE_MAE_SCORE:.2f} months")
    print(f"-> Multi-Input Model Val MAE : {best_val_mae:.2f} months")

    # Baseline ile yeni model arasındaki farkın hesaplanması
    difference_mae = BASELINE_MAE_SCORE - best_val_mae

    # Yeni model baseline'dan daha iyi performans gösteriyorsa
    if difference_mae > 0:

        print(
            f"\n[SUCCESS] Multi-Input architecture improved "
            f"the model by {difference_mae:.2f} months."
        )

        print(
            "[SUCCESS] Acceptance criteria satisfied. "
            "Ready to close the issue."
        )

    # Yeni model baseline'dan daha kötü performans gösteriyorsa
    elif difference_mae < 0:

        print(
            f"\n[WARNING] Model is "
            f"{abs(difference_mae):.2f} months worse "
            f"than the baseline."
        )

    # Performanslar eşitse
    else:

        print(
            "\n[INFO] Multi-Input model achieved "
            "the same performance as the baseline."
        )

# val_mae metriği eğitim geçmişinde bulunamazsa
else:

    print(
        "[WARNING] Validation MAE metric not found "
        "in training history."
    )

print("-" * 50)