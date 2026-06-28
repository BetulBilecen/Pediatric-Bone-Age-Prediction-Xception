# --------------------------------------------------------------
# KÜTÜPHANELERİN YÜKLENMESİ (IMPORT LIBRARIES)
# --------------------------------------------------------------
import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam

# --------------------------------------------------------------
# YARDIMCI FONKSİYONLAR (UTILITY FUNCTIONS)
# --------------------------------------------------------------
def build_multi_input_model(img_size):
    # 1. Girdi katmanı
    image_input = Input(shape=(img_size[0], img_size[1], 3), name="image_input")

    base_model = tf.keras.applications.xception.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input
    )
    base_model.trainable = True  # Fine-tuning için açık bırakıyoruz

    x = GlobalAveragePooling2D()(base_model.output)

    # 2. Girdi Kolu: Tabular Cinsiyet Verisi
    gender_input = Input(shape=(1,), name="gender_input")
    gender_branch = Dense(16, activation="relu")(gender_input)

    # cinsiyet ve görüntüden gelen verileri birleştirilmesi
    combined = Concatenate()([x, gender_branch])
    fc = Dense(32, activation="relu")(combined)
    output = Dense(1, activation="linear", name="boneage_output")(fc)

    # Model girdilerini ve çıktılarını tanımlama
    model = Model(inputs=[image_input, gender_input], outputs=output)
    model.compile(loss="mse",optimizer=Adam(learning_rate=0.0001),metrics=["mae"]
    )
    return model