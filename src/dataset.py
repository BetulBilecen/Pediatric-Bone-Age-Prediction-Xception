# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import pandas as pd
import os
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

    # Veriyi bölme: %80 eğitim, %10 doğrulama, %10 test
    df_train, df_val = train_test_split(df_training, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)

    # Veri arttırma (Data Augmentation)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Eğitim ve doğrulama generator'larının hazırlanması
    train_gen = multi_input_generator(train_datagen, df_train, img_size, batch_size, is_training=True)
    val_gen = multi_input_generator(val_test_datagen, df_val, img_size, batch_size, is_training=False)

    return train_gen, val_gen, len(df_train), len(df_val), df_test