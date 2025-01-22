import random
import tensorflow as tf
from trdg.generators import GeneratorFromStrings
from PIL import Image
import numpy as np
from pathlib import Path
import json

if __name__ == "__main__":
    # Полный набор символов: буквы, цифры, знаки препинания
    alphabet = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        " ,.!?-—;:"
    )


    # Генерируем слова и фразы из символов
    def generate_random_texts(count=500):
        import random
        texts = []
        for _ in range(count):
            length = random.randint(5, 20)  # Длина строки от 5 до 20 символов
            text = "".join(random.choices(alphabet, k=length))
            if text:  # Проверка, что строка не пустая
                texts.append(text)
        return texts


    russian_texts = generate_random_texts(100)  # Генерация 100 случайных строк

    # Путь к шрифтам
    font_dir = Path('resources', 'russian_fonts')
    fonts = [str(font_path) for font_path in font_dir.glob("*.ttf")]

    if not fonts:
        raise FileNotFoundError(f"В папке {font_dir} не найдено файлов шрифтов (*.ttf)")

    # Папка для сохранения изображений
    output_dir = Path('resources', 'data', 'train_images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Путь для сохранения файла с метками
    labels_path_json = output_dir / "labels.json"
    labels = []

    # Генератор изображений
    generator = GeneratorFromStrings(
        strings=russian_texts,
        count=len(russian_texts),
        fonts=fonts,
        size=62,
        skewing_angle=85,  # Устанавливаем диапазон углов наклона (максимум ±85 градусов)
        random_skew=True,
        blur=1,  # Увеличиваем размытие
        background_type=2,  # Тип фона: 0 — белый, 1 — случайный цвет, 2 — случайное изображение
    )

    for i, (image, text) in enumerate(generator):
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        if not text:
            print(f"Warning: Empty text for image_{i + 1:04d}")
            continue

        image_filename = f"image_{i + 1:04d}.jpg"
        image_path = output_dir / image_filename
        pil_image.save(image_path)

        # Сохраняем полный путь и метку
        labels.append({"image": str(image_path), "label": text})
        print(f"Сохранено: {image_path} с меткой '{text}'")

    with open(labels_path_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"\nМетки сохранены в файл: {labels_path_json}")
