import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


class DataGenerator(Sequence):
    def __init__(self, csv_file, image_dir, charset, batch_size=32, input_shape=(128, 32, 1)):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.charset = charset
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.blank_idx = len(charset)  # Индекс для пустого символа (blank)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        labels = []
        input_lengths = []
        label_lengths = []

        for _, row in batch_data.iterrows():
            # Формируем полный путь к изображению
            img_path = self.image_dir / row['image']

            # Чтение изображения
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # Проверка, если изображение не найдено или не может быть загружено
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue

            # Изменяем размер изображения
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))  # (128, 32)
            image = image / 255.0  # Нормализация
            image = np.expand_dims(image, axis=-1)  # Добавляем канал, чтобы получилось (128, 32, 1)

            # Проверка формы изображения
            print(f"Resized image shape: {image.shape}")

            # Преобразование текста в последовательность индексов
            label = [self.charset.index(c) if c in self.charset else self.blank_idx for c in row['label']]
            labels.append(label)

            # Вычисление input_lengths
            input_length = max(4, image.shape[1] // 4)  # Обеспечиваем, что input_length >= 4
            input_lengths.append(input_length)

            # Длина метки
            label_lengths.append(len(label))

            # Добавление изображения в список
            images.append(image)

        # Преобразуем в массивы
        images = np.array(images, dtype=np.float32)
        labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post', value=self.blank_idx,
                                                               dtype=np.int32)
        label_lengths = np.array(label_lengths, dtype=np.int32)
        input_lengths = np.array(input_lengths, dtype=np.int32)

        # Отладка: проверим формы
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Input lengths shape: {input_lengths.shape}")
        print(f"Label lengths shape: {label_lengths.shape}")

        # Возвращаем данные для CTC
        return images, labels, input_lengths, label_lengths
