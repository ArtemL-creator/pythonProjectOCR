import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import StringLookup
from pathlib import Path

# Параметры
validation_split = 0.2  # Доля данных для тестовой выборки

# Путь к JSON-файлу
json_path = Path("resources/data/train_images/labels.json")

# Чтение и парсинг JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Извлечение данных
image_paths = [entry["image"] for entry in data]
labels = [entry["label"] for entry in data]

# Перемешивание данных
np.random.seed(42)
indices = np.arange(len(image_paths))
np.random.shuffle(indices)

image_paths = np.array(image_paths)[indices]
labels = np.array(labels)[indices]

# Определение индексов для разделения
split_idx = int(len(image_paths) * (1 - validation_split))
train_image_paths = image_paths[:split_idx]
train_labels = labels[:split_idx]

validation_image_paths = image_paths[split_idx:]
validation_labels = labels[split_idx:]

# Создание маппинга символов
alphabet = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    " ,.!?-—;:"
)

char_to_num = StringLookup(vocabulary=list(alphabet), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# Размеры изображения
image_height = 32
image_width = 128


# Преобразование изображений
def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize_with_pad(image, img_size[1], img_size[0])
    image = tf.cast(image, tf.float32) / 255.0
    return image


# Длина самой длинной метки
max_len = max(len(label) for label in labels)


# Преобразование меток
def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    pad_amount = max_len - tf.shape(label)[0]
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=0)
    return label


def process_image_label(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


batch_size = 16


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(process_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_ds = prepare_dataset(train_image_paths, train_labels)
validation_ds = prepare_dataset(validation_image_paths, validation_labels)

for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    for i in range(len(images)):
        # Восстанавливаем путь к изображению
        image_path = train_image_paths[i]  # Путь к текущему изображению

        # Восстанавливаем метку, убирая padding
        label = ''.join(
            [num_to_char(ch).numpy().decode('utf-8') for ch in labels[i] if ch != 0]
        )

        print(f"Image Path: {image_path}")
        print(f"Label: {label}")

for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    # Количество изображений для отображения
    num_images_to_show = min(len(images), 16)  # Ограничим до 16 изображений
    num_cols = 4  # Количество столбцов в сетке
    num_rows = (num_images_to_show + num_cols - 1) // num_cols  # Вычисляем строки

    _, ax = plt.subplots(num_rows, num_cols, figsize=(15, 8))

    for i in range(num_images_to_show):
        # Получаем изображение
        img = images[i].numpy().squeeze()  # Убираем лишний размер (1 канал)

        # Получаем путь и метку
        image_path = train_image_paths[i]
        label = ''.join(
            [num_to_char(ch).numpy().decode('utf-8') for ch in labels[i] if ch != 0]
        )

        # Вывод изображения
        ax[i // num_cols, i % num_cols].imshow(img, cmap="gray")
        ax[i // num_cols, i % num_cols].set_title(f"Label: {label}\nPath: {image_path}", fontsize=8)
        ax[i // num_cols, i % num_cols].axis("off")

    # Убираем лишние оси
    for j in range(num_images_to_show, num_rows * num_cols):
        ax[j // num_cols, j % num_cols].axis("off")

    plt.tight_layout()
    plt.show()