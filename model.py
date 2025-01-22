import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image

# Параметры
json_path = Path("resources/data/train_images/labels.json")
image_height = 32  # Высота входного изображения для модели
image_width = 128  # Ширина входного изображения для модели
batch_size = 16  # Размер батча
validation_split = 0.2  # Доля данных для тестовой выборки

# Алфавит (должен соответствовать набору символов в данных)
alphabet = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    " ,.!?-—;:"
)
char_to_index = {char: idx + 1 for idx, char in enumerate(alphabet)}  # +1 для 0-padding
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Функция для чтения JSON
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Разделение на обучающую и тестовую выборки
def split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test

data = load_data(json_path)
train_data, test_data = split_data(data, test_size=validation_split)
print(f"Данные разделены: {len(train_data)} на обучение, {len(test_data)} на тест")

# Функция преобразования текста в индексы
def text_to_indices(text):
    return [char_to_index[char] for char in text if char in char_to_index]

# Функция загрузки изображения и приведения к нужному размеру
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Градации серого
    image = image.resize((image_width, image_height))  # Масштабирование
    image = np.array(image, dtype=np.float32) / 255.0  # Нормализация
    return image.T  # Транспонируем для CTC (ширина x высота)

# Генератор данных
def data_generator(data, batch_size):
    while True:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            images, labels, input_lengths, label_lengths = [], [], [], []

            for item in batch:
                # Обработка изображения
                image = preprocess_image(item["image"])
                images.append(image)

                # Текст в индексы
                label = text_to_indices(item["label"])
                labels.append(label)

                # Длины входа и меток
                input_lengths.append(image_width // 4)  # Пропорционально сверткам
                label_lengths.append(len(label))

            # Паддинг для меток
            max_label_length = max(label_lengths)
            padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
                labels, maxlen=max_label_length, padding="post", value=0
            )

            # Преобразование в numpy массивы
            images = np.array(images).reshape(-1, image_width, image_height, 1)
            input_lengths = np.array(input_lengths).reshape(-1, 1)
            label_lengths = np.array(label_lengths).reshape(-1, 1)

            yield (
                {
                    "image": images,
                    "label": padded_labels,
                    "input_length": input_lengths,
                    "label_length": label_lengths,
                }
                # ,
                # np.zeros(len(images)),  # Заглушка для CTC Loss
            )


def build_ctc_model():
    input_img = tf.keras.layers.Input(shape=(image_width, image_height, 1), name="image")
    labels = tf.keras.layers.Input(name="label", shape=(None,), dtype="int32")
    input_length = tf.keras.layers.Input(name="input_length", shape=(1,), dtype="int32")
    label_length = tf.keras.layers.Input(name="label_length", shape=(1,), dtype="int32")

    # Сверточные слои
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Преобразование для RNN
    new_shape = (image_width // 4, (image_height // 4) * 64)
    x = tf.keras.layers.Reshape(target_shape=new_shape)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # RNN
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

    # Выходной слой
    output = tf.keras.layers.Dense(len(alphabet) + 1, activation="softmax")(x)

    # Определение CTC Loss
    ctc_loss = tf.keras.layers.Lambda(
        lambda inputs: tf.keras.backend.ctc_batch_cost(*inputs),
        name="ctc_loss"
    )([labels, output, input_length, label_length])

    model = tf.keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=ctc_loss
    )

    prediction_model = tf.keras.models.Model(inputs=input_img, outputs=output)
    return model, prediction_model

if __name__ == "__main__":
    # Создание модели
    model, prediction_model = build_ctc_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={"ctc_loss": lambda y_true, y_pred: y_pred},  # Используем заглушку
    )

    # Создание генераторов
    train_gen = data_generator(train_data, batch_size)
    batch = next(train_gen)
    for key, value in batch[0].items():
        print(f"{key}: {value.shape}")
    val_gen = data_generator(test_data, batch_size)

    # Обучение
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_data) // batch_size,
        validation_steps=len(test_data) // batch_size,
        epochs=20,
    )

    # Сохранение обученной модели
    model_save_path = "ctc_model.h5"
    prediction_model_save_path = "ctc_prediction_model.h5"
    model.save(model_save_path)
    prediction_model.save(prediction_model_save_path)
    print(f"Модель сохранена: {model_save_path}, {prediction_model_save_path}")
