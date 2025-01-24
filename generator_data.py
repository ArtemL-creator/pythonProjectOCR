import random
from trdg.generators import GeneratorFromStrings
from PIL import Image
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Полный набор символов: буквы, цифры, знаки препинания
    alphabet = (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        " ,.!?—«»"
    )


    # Генерируем слова и фразы из символов
    def generate_random_texts(count=500):
        texts = []
        for _ in range(count):
            length = random.randint(3, 8)  # Длина строки
            text = "".join(random.choices(alphabet, k=length))
            if text:  # Проверка, что строка не пустая
                texts.append(text)
        return texts


    russian_texts = generate_random_texts(10000)  # Генерация случайных строк

    # Путь к шрифтам
    font_dir = Path('resources', 'russian_fonts')
    fonts = [str(font_path) for font_path in font_dir.glob("*.ttf")]

    if not fonts:
        raise FileNotFoundError(f"В папке {font_dir} не найдено файлов шрифтов (*.ttf)")

    # Папка для сохранения изображений
    output_dir = Path('resources', 'data', 'train_images_for_ts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Генератор изображений
    generator = GeneratorFromStrings(
        strings=russian_texts,
        count=len(russian_texts),
        fonts=fonts,
        size=42,
        skewing_angle=15,  # Устанавливаем диапазон углов наклона
        random_skew=True,
        blur=random.randrange(1, 2),  # Увеличиваем размытие
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

        # Сохраняем изображение
        image_filename = f"image_{i + 1:04d}.tif"
        image_path = output_dir / image_filename
        pil_image.save(image_path)

        # Сохраняем метку в .gt.txt
        gt_file_path = output_dir / f"image_{i + 1:04d}.gt.txt"
        with open(gt_file_path, "w", encoding="utf-8") as gt_file:
            gt_file.write(text)

        # Генерация .box файла
        box_file_path = output_dir / f"image_{i + 1:04d}.box"
        with open(box_file_path, "w", encoding="utf-8") as box_file:
            width, height = pil_image.size
            char_width = width // len(text)  # Предполагаемая ширина символа
            x_start = 0

            for char in text:
                x_end = x_start + char_width
                # Записываем строку: символ, x_start, y_start, x_end, y_end, номер строки (0)
                box_line = f"{char} {x_start} 0 {x_end} {height} 0\n"
                box_file.write(box_line)
                x_start = x_end

        print(f"Сохранено: {image_path}, .gt.txt, .box с меткой '{text}'")
