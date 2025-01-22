import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import json
from pathlib import Path
import pickle

# Пути к данным
image_dir = Path('resources', 'data', 'train_images')
json_path = image_dir / "labels.json"

# Чтение меток
with open(json_path, 'r', encoding='utf-8') as f:
    labels_data = json.load(f)

# Алфавит
alphabet = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    " ,.!?-—;:"
)
num_classes = len(alphabet) + 1  # +1 для пустого символа (для CTC)

# Словарь для маппинга символов в индексы
char_to_index = {char: idx + 1 for idx, char in enumerate(alphabet)}
char_to_index[''] = 0  # Пустой символ

# Функция для кодирования меток
def encode_labels(text):
    return [char_to_index[char] for char in text]

def pad_labels(labels, max_label_length):
    return [label + [0] * (max_label_length - len(label)) for label in labels]

# Загрузка данных
def load_data():
    images = []
    labels = []
    label_lengths = []  # Для CTC нам нужны длины меток

    for item in labels_data:
        image_path = image_dir / item['image']
        label = item['label']

        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 32))  # Изменяем размер изображения
        image = np.array(image) / 255.0  # Нормализация
        images.append(image)

        # Кодирование меток
        encoded_label = encode_labels(label)
        labels.append(encoded_label)
        label_lengths.append(len(encoded_label))  # Записываем длину метки

    images = np.array(images)

    # Преобразуем метки в формат CTC
    max_label_length = max(label_lengths)  # Вычисляем максимальную длину метки
    labels = pad_labels(labels, max_label_length)
    labels = np.array(labels)
    label_lengths = np.array(label_lengths)

    return images, labels, label_lengths, max_label_length


def ctc_loss_lambda_func(y_true, y_pred):
    """
    Реализация CTC Loss с использованием TensorFlow.
    """
    # Преобразование длины меток
    label_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)

    # Получение batch размера
    batch_size = tf.shape(y_pred)[0]

    # Длина предсказаний (количество временных шагов)
    input_length = tf.fill([batch_size], tf.shape(y_pred)[1])

    # Преобразование y_true к int32
    y_true = tf.cast(y_true, dtype=tf.int32)

    # Расчет CTC Loss
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_lengths,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=0  # Используем 0 как blank символ
    )
    return tf.reduce_mean(loss)

# Создание модели CTC
def build_ctc_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name='input_1')

    # Сверточные слои
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Преобразование формы
    x = layers.Reshape((-1, 64))(x)

    # Скрытые слои
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Выходной слой без активации (логиты)
    output = layers.Dense(num_classes)(x)

    model = models.Model(inputs, output)

    return model

# Загрузка данных
images, labels, label_lengths, max_label_length = load_data()

# Длины входных последовательностей
time_steps = 8  # Количество временных шагов модели
input_lengths = np.full((len(images),), time_steps, dtype=np.int32)

# Создаем словарь данных
data = {
    'input_1': images,          # Входные изображения
    'labels': labels,           # Метки
    'input_lengths': input_lengths,  # Длина входных последовательностей
    'label_lengths': label_lengths   # Длина меток
}

# Пример параметров
input_shape = (32, 128, 3)  # Размеры изображений

# Создание модели
model = build_ctc_model(input_shape, num_classes)

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Уменьшенная скорость обучения
    loss=ctc_loss_lambda_func
)

# Вывод структуры модели
model.summary()

# Колбэк для управления скоростью обучения
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)

# Тренировка модели
model.fit(
    data,
    labels,
    batch_size=16,
    epochs=30,
    callbacks=[reduce_lr]
)

# Сохранение обученной модели и алфавита
model.save("ocr_train_model_ctc.keras")
with open("alphabet.pkl", "wb") as f:
    pickle.dump(alphabet, f)

# Проверка предсказаний
y_pred = model.predict(images[:5])
decoded_sequences = tf.keras.backend.ctc_decode(y_pred, input_lengths[:5])[0][0].numpy()

for i, seq in enumerate(decoded_sequences):
    decoded_text = "".join([alphabet[char - 1] for char in seq if char > 0])  # Преобразуем индексы в текст
    print(f"Предсказание {i + 1}: {decoded_text}")

    # Отображаем изображение
    plt.imshow(images[i])  # Показываем изображение
    plt.title(f"Предсказание: {decoded_text}")
    plt.axis('off')  # Убираем оси
    plt.show()
