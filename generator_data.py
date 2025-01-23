from trdg.generators import GeneratorFromStrings
from PIL import Image
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Полный набор символов: буквы, цифры, знаки препинания
    alphabet = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        " ,.!?—«»"
    )


    # Генерируем слова и фразы из символов
    def generate_random_texts(count=500):
        import random
        texts = []
        for _ in range(count):
            length = random.randint(3, 6)  # Длина строки
            text = "".join(random.choices(alphabet, k=length))
            if text:  # Проверка, что строка не пустая
                texts.append(text)
        return texts


    russian_texts = generate_random_texts(1000)  # Генерация 100 случайных строк

    # Путь к шрифтам
    font_dir = Path('resources', 'russian_fonts')
    fonts = [str(font_path) for font_path in font_dir.glob("*.ttf")]

    if not fonts:
        raise FileNotFoundError(f"В папке {font_dir} не найдено файлов шрифтов (*.ttf)")

    # Папка для сохранения изображений
    output_dir = Path('resources', 'data', 'train_images_for_ts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Путь для сохранения файла с метками
    labels_path_json = output_dir / "labels.json"
    labels = []

    # Генератор изображений
    generator = GeneratorFromStrings(
        strings=russian_texts,
        count=len(russian_texts),
        fonts=fonts,
        size=42,
        skewing_angle=15,  # Устанавливаем диапазон углов наклона
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

        image_filename = f"image_{i + 1:04d}.tif"
        image_path = output_dir / image_filename
        pil_image.save(image_path)

        # Сохраняем метку в .gt.txt
        with open(output_dir / f"image_{i + 1:04d}.gt.txt", "w", encoding="utf-8") as gt_file:
            gt_file.write(text)

        print(f"Сохранено: {image_path} с меткой '{text}'")
