# ----------------------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # GUI backend'i devre dışı bırakır, tkinter hatasını önler
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from matplotlib.image import imread

import warnings
warnings.filterwarnings("ignore")   # Uyarı mesajlarını görmezden gelir, gereksiz kalabalığı önlemek için

# ----------------------------------------------------------------------------
# 2. VERİ HAZIRLAMA VE ETİKETLEME (DATA PREPARATION AND LABELING)
# ----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
df_direction = os.path.join(current_dir, "bonage_dataset")

#  A. EĞİTİM VERİSİ
# ----------------------------------------------------------------------------
df_training = pd.read_csv(os.path.join(df_direction, "boneage-training-dataset.csv"))

# Dosya yollarının hazırlanması
df_training["path"] = df_training["id"].map(
    lambda x: os.path.join(df_direction, "boneage-training-dataset", "{}.png".format(x))
)
df_training["image_path"] = df_training["id"].map(lambda y: "{}.png".format(y))

# Cinsiyet bilgilerinin düzenlenmesi
df_training["gender"] = df_training["male"].map(lambda z: "male" if z else "female")           # male -> True/False yerine gender sütunu eklendi
df_training["gender_encoded"] = df_training["gender"].map(lambda z: 1 if z == "male" else 0)   # Kadınlar -> 0, Erkekler -> 1

# Veri analizi için kemik yaşının 10 eşit genişlikteki aralığa bölünmesi
df_training["boneage_category"] = pd.cut(df_training["boneage"], 10)

# Yaş verisinin normalize edilmesi
boneage_std = 2 * df_training["boneage"].std()
boneage_mean = df_training["boneage"].mean()
df_training["norm_age"] = (df_training["boneage"] - boneage_mean) / boneage_std

#  B. TEST VERİSİ
# ----------------------------------------------------------------------------
df_test_raw = pd.read_csv(os.path.join(df_direction, "boneage-test-dataset.csv"))

df_test_raw["path"] = df_test_raw["Case ID"].map(
    lambda x: os.path.join(df_direction, "boneage-test-dataset", "{}.png".format(x))
)
df_test_raw["image_path"] = df_test_raw["Case ID"].map(lambda y: "{}.png".format(y))

df_test_raw["gender"] = df_test_raw["Sex"].map(lambda z: "male" if z == "M" else "female")
df_test_raw["gender_encoded"] = df_test_raw["gender"].map(lambda z: 1 if z == "male" else 0)

# ----------------------------------------------------------------------------
# C. DOSYA BÜTÜNLÜĞÜ KONTROLÜ (IMAGE VALIDATION)
# ----------------------------------------------------------------------------
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        img.close()
        return True
    except (IOError, OSError):
        return False

print(f"Başlangıç eğitim verisi: {len(df_training)} görüntü")

df_training["is_valid"] = df_training["path"].apply(is_valid_image)
df_training = df_training[df_training["is_valid"]].drop(columns=["is_valid"])

print(f"Doğrulama sonrası eğitim verisi: {len(df_training)} görüntü")

# ----------------------------------------------------------------------------
# D. GÖRSELLEŞTİRME — Rastgele Örnek Görüntü
# ----------------------------------------------------------------------------
random_index = np.random.randint(0, len(df_training))
single_img_path = df_training["path"].iloc[random_index]
imgs = imread(single_img_path)
plt.figure()
plt.imshow(imgs, cmap="gray")
plt.axis("off")
plt.title(f"Örnek Görüntü (Kemik Yaşı: {df_training['boneage'].iloc[random_index]} ay)")
plt.tight_layout()

# Görüntüyü kaydet
images_dir = os.path.join(current_dir, "Images")
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, "sample_xray.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# 3. ÖN İŞLEME VE VERİ ARTIRMA (PREPROCESSING & DATA AUGMENTATION)
# ----------------------------------------------------------------------------

# Veri setini ayırma: %80 eğitim, %10 doğrulama, %10 test
df_train, df_val = train_test_split(df_training, test_size=0.2, random_state=42, shuffle=True)
df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42, shuffle=True)


# A. VERİ ARTIRMA (DATA AUGMENTATION)
# ----------------------------------------------------------------------------
data_augmentation = dict(
    rotation_range=180,         # Resimleri rastgele 180 dereceye kadar döndürme
    zoom_range=0.25,            # Resimleri rastgele yakınlaştırma veya uzaklaştırma
    brightness_range=[0.8, 1.2],# Parlaklık aralığı — çok geniş aralık kemik uçlarındaki detayları kaybettiriyordu
    height_shift_range=0.2,     # Resimleri y ekseninde rastgele kaydırma
    horizontal_flip=True,       # Resimleri rastgele yatay olarak çevirme
    shear_range=0.05,           # Resimlere rastgele eğme (shear) işlemi uygulama
    fill_mode="nearest"         # Dönüşüm sonrası oluşan boşlukları en yakın piksellerle doldurma
)

train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,    # Xception'a özel ön işleme: piksel değerlerini [-1,1] aralığına çeker
    **data_augmentation
)
test_val_generator = ImageDataGenerator(preprocessing_function=preprocess_input)   # Augmentation yok, sadece normalize

# B. VERİ AKIŞININ YAPILANDIRILMASI (FLOW FROM DATAFRAME)
# ----------------------------------------------------------------------------
img_size = (128, 128)
batch_size = 32

train_data = train_generator.flow_from_dataframe(
    dataframe= df_train,
    x_col= "path",
    y_col= "boneage",
    seed= 42,
    shuffle= True,
    class_mode= "other",
    color_mode= "rgb",
    target_size= img_size
)

valid_data = test_val_generator.flow_from_dataframe(
    dataframe= df_val,
    x_col= "path",
    y_col= "boneage",
    seed= 42,
    shuffle= False,
    class_mode= "other",
    color_mode= "rgb",
    batch_size= batch_size,
    target_size= img_size
)

test_data = test_val_generator.flow_from_dataframe(
    dataframe=df_test,
    x_col= "path",
    y_col= "boneage",
    seed= 42,
    shuffle= False,
    class_mode= "other",
    color_mode= "rgb",
    batch_size= batch_size,
    target_size= img_size
)

# Model değerlendirmesi için toplu test verisinin tek seferde belleğe alınması
X_test, y_test = next(test_val_generator.flow_from_dataframe(
    df_test,
    x_col= "path",
    y_col= "boneage",
    batch_size= batch_size * 10,
    class_mode= "other",
    color_mode= "rgb",
    target_size= img_size
))

# ----------------------------------------------------------------------------
# 4. TRANSFER LEARNING: XCEPTION + FINE TUNING
# ----------------------------------------------------------------------------
base_model = tf.keras.applications.xception.Xception(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,          # Sınıflandırma katmanları dahil edilmedi
    weights="imagenet"          # ImageNet ağırlıkları ile başlatıldı
)
base_model.trainable = True     # Fine-tuning: tüm katmanlar eğitilebilir

new_model = Sequential([
    base_model,
    GlobalMaxPooling2D(),
    Dense(5),
    Activation("relu"),
    Dense(1, activation="linear")       # Regresyon çıkışı: tek sayısal değer (ay bazında yaş)
])

new_model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["mae", "mse"]
)

new_model.summary()

# ----------------------------------------------------------------------------
# 5. MODEL EĞİTİMİ (MODEL TRAINING)
# ----------------------------------------------------------------------------
callback = EarlyStopping(
    monitor="val_loss",
    patience=5,                     # 5 epoch iyileşme olmazsa durdur
    restore_best_weights=True       # En iyi checkpoint'e geri dön
)

print("\n[BİLGİ] Eğitim başlıyor...")
history = new_model.fit(
    train_data,
    epochs=25,
    validation_data=valid_data,
    batch_size=batch_size,
    callbacks=[callback]
)

# ----------------------------------------------------------------------------
# 6. MODEL DEĞERLENDİRME (MODEL EVALUATION)
# ----------------------------------------------------------------------------
print("\n--- Model Performans Özeti ---")
results = new_model.evaluate(X_test, y_test, batch_size=16, verbose=1)
print(f"Test MSE : {results[0]:.2f}")
print(f"Test MAE : {results[1]:.2f} ay")

# Tahminler
y_pred = new_model.predict(X_test, batch_size=16, verbose=1)

# Yaşa göre sıralı 6 örnek seç
sorted_indices = np.argsort(y_test)
ord_ = sorted_indices[np.linspace(0, len(sorted_indices) - 1, 6).astype(int)]

fig, ax = plt.subplots(1, 6, figsize=(20, 5))

for (i, _ax) in zip(ord_, ax.flatten()):
    _ax.imshow((X_test[i] + 1) / 2)    # preprocess_input [-1,1] -> [0,1] aralığına geri alınır
    _ax.set_title(
        f"Real Age: {y_test[i]:.0f}\nPred: {y_pred[i][0]:.1f}",
        fontsize=12, fontweight="bold"
    )
    _ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "boneage_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nTahmin görseli kaydedildi: {os.path.join(images_dir, 'boneage_predictions.png')}")