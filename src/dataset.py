# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input

# --------------------------------------------------------------
# YARDIMCI FONKSİYONLAR (UTILITY FUNCTIONS)
# --------------------------------------------------------------

# Görüntü dosyası bozuksa False döndürecek şekilde düzenleme
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, OSError):
        return False

# Multi-Input model için görüntü ve cinsiyet verilerini birlikte üreten generator
def multi_input_generator(generator, dataframe, img_size, batch_size, is_training=True):
    gen = generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col="path",
        y_col=["gender_encoded", "boneage"],  # İki nümerik sütunu birlikte çıkarıyoruz
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        color_mode="rgb",
        shuffle=is_training,
        seed=42
    )

    while True:
        X_batch, y_batch = next(gen)    # X görüntü matrisi
        yield [X_batch, y_batch[:, 0]], y_batch[:, 1] # y_batch[:, 1]: kemik yaşı, y_batch[:, 0]: cinsiyet bilgisi


# Maskeleme işlemi: Etiketi siyah noktalar ile boyama
def mask_image(image_path):
    _img = cv2.imread(str(image_path))

    if _img is None:    return None
    gray = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)

    # Etiket belirginleştirme
    _, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    # Şekil sınırlarını bulma
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # _img 3 kanallı (BGR) olduğu için shape üç değer döndürür, kanal sayısını atıyoruz
    _height, _width = _img.shape[:2]
    img_area = _height * _width

    for _contur in contours:
        # Kontur etrafında hayali bir kutu çizme
        x,y,nw,nh = cv2.boundingRect(_contur)
        box_area = nw * nh

        # Bulunan şekil tüm resmin %1-%15'i arasındaysa (marker/etiket boyutu varsayımı)
        if (img_area * 0.01) < box_area and box_area < (img_area * 0.15):
            _img[y: y + nh, x: x + nw] = 0

    # Tüm konturlar kontrol edildikten sonra tek seferde resize ediliyor
    return cv2.resize(_img, (128, 128))


# Maskelenmiş görüntüleri diske kaydetme (her epoch'ta tekrar maskelemek yerine bir kerelik işlem)
def create_masked_dataset(df_training, output_dir):
    os.makedirs(output_dir, exist_ok= True)
    masked_paths = []

    for idx, row in df_training.iterrows():
        masked_img = mask_image(row["path"])

        # Maskeleme başarısızsa (bozuk görüntü vs.) path'i None bırak, sonra filtrelenecek
        if masked_img is None:
            masked_paths.append(None)
            continue

        # output_dir kullanılıyor (önceki output_path hatası düzeltildi)
        output_path = os.path.join(output_dir, f"{row['id']}.png")
        cv2.imwrite(output_path, masked_img)
        masked_paths.append(output_path)

    # Orijinal path'lerin yerine maskelenmiş görüntülerin path'lerini yazma
    df_training["path"] = masked_paths
    df_training = df_training[df_training["path"].notna()]

    return df_training


# Veri yükleme, temizleme, bölme ve veri artırma işlemleri
def prepare_data(df_direction, img_size, batch_size):
    df_training = pd.read_csv(os.path.join(df_direction, "boneage-training-dataset.csv"))

    # Dosya yollarını oluşturma
    df_training["path"] = df_training["id"].map(lambda x: os.path.join(df_direction, "boneage-training-dataset", f"{x}.png"))

    # Cinsiyet bilgisini sayısal formata dönüştürme ( 0/1 encoding)
    df_training["gender_encoded"] = df_training["male"].map(lambda x: 1 if x else 0)

    # Bozuk resimleri temizleme
    df_training["is_valid"] = df_training["path"].apply(is_valid_image)
    df_training = df_training[df_training["is_valid"]].drop(columns=["is_valid"])

    # Etiket/marker maskeleme: modelin sabit bir köşeye kilitlenmesini önlemek için
    masked_dir = os.path.join(df_direction, "boneage-masked-dataset")
    df_training = create_masked_dataset(df_training, masked_dir)

    # Veriyi bölme: %80 eğitim, %10 doğrulama, %10 test
    df_train, df_val = train_test_split(df_training, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)

    # Veri arttırma (Data Augmentation)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,           # Görüntüyü rastgele -20° ile +20° arasında döndürür
        height_shift_range= 0.15,      # Görüntüyü dikey kaydırma
        width_shift_range = 0.15,      # Görüntüyü yatay kaydırma
        fill_mode= "nearest",        # Kaydırmalarda boş kalan yerleri 0 (siyah) yapar
        zoom_range=0.15,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Eğitim ve doğrulama generator'larının hazırlanması
    train_gen = multi_input_generator(train_datagen, df_train, img_size, batch_size, is_training=True)
    val_gen = multi_input_generator(val_test_datagen, df_val, img_size, batch_size, is_training=False)

    return train_gen, val_gen, len(df_train), len(df_val), df_test