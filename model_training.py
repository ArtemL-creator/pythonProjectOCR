import tensorflow as tf
from keras.api.models import Model
from keras.api.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Пути к данным
    data_dir = Path('resources', 'data', 'train_images')
    labels_file = data_dir / "labels.csv"

    # Чтение меток
    labels_df = pd.read_csv(labels_file)

    # Разделение на тренировочный и валидационный наборы
    train_df = labels_df.sample(frac=0.8, random_state=42)  # 80% для обучения
    val_df = labels_df.drop(train_df.index)  # Остальные 20% для валидации

    # Создание генераторов данных
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    # Подготовка данных
    # Создание генераторов данных
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=data_dir,
        x_col="image",
        y_col="label",
        target_size=(32, 128),  # Размеры входного изображения
        color_mode="grayscale",  # Используем grayscale
        class_mode="sparse",  # Для многоклассовой классификации
        batch_size=32,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=data_dir,
        x_col="image",
        y_col="label",
        target_size=(32, 128),
        color_mode="grayscale",  # Используем grayscale
        class_mode="sparse",
        batch_size=32,
        shuffle=False
    )

    # Параметры модели
    input_shape = (32, 128, 3)  # Размеры изображения
    num_classes = len(train_generator.class_indices)  # Количество классов (символов)

    # Архитектура модели
    inputs = Input(shape=input_shape)

    # Сверточные слои
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Преобразование для RNN
    x = Reshape(target_shape=(-1, 64))(x)

    # Рекуррентные слои
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)

    # Выходной слой
    outputs = Dense(num_classes, activation="softmax")(x)

    # Создание модели
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Обучение модели
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    # Сохранение модели
    model.save("ocr_model.h5")