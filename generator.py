import random
from trdg.generators import GeneratorFromStrings
from PIL import Image
import numpy as np
from pathlib import Path
import csv  # Для сохранения меток

if __name__ == "__main__":
    # Тексты для генерации
    russian_texts = [
        "Автомобиль Москвич", "Быстрый урок", "Великая наука", "Гениальная идея",
        "День рождения", "Единство природы", "Жаркий летний день", "Загадка века",
        "Идеальный баланс", "Йога для тела", "Кулинарный шедевр", "Летний дождь",
        "Магический лес", "Небо без облаков", "Океанская глубина", "Путеводная звезда",
        "Радуга после дождя", "Сильный ветер", "Теплый огонь", "Удивительные приключения",
        "Философия жизни", "Химия природы", "Цветочная поляна", "Чистая вода",
        "Шумный город", "Щедрый урожай", "Энергия движения", "Южный ветер",
        "Яркий рассвет", "Преображение мира", "Ритм жизни", "Секреты успеха",
        "Эмоции и чувства", "Будущее технологий", "Тайна вселенной", "Далёкие звезды",
        "Гармония души", "Сила мысли", "Секреты древних знаний", "Техники медитации",
        "Звездный вечер", "Путь к вершинам"
    ]

    # Путь к шрифтам
    font_dir = Path('resources', 'russian_fonts')
    fonts = [str(font_path) for font_path in font_dir.glob("*.ttf")]

    if not fonts:
        raise FileNotFoundError(f"В папке {font_dir} не найдено файлов шрифтов (*.ttf)")

    # Папка для сохранения изображений
    output_dir = Path('resources', 'data', 'train_images')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Путь для сохранения файла с метками
    labels_path = output_dir / "labels.csv"

    # Генератор изображений
    generator = GeneratorFromStrings(
        strings=russian_texts,
        count=350,
        fonts=fonts,
        size=62,
        skewing_angle=85,  # Устанавливаем диапазон углов наклона (максимум ±85 градусов)
        random_skew=True,
        blur=1,  # Увеличиваем размытие
        background_type=2,  # Тип фона: 0 — белый, 1 — случайный цвет, 2 — случайное изображение
    )

    # Открываем CSV для записи меток
    with open(labels_path, mode='w', newline='', encoding='utf-8') as labels_file:
        csv_writer = csv.writer(labels_file)
        csv_writer.writerow(["image", "label"])  # Заголовок для CSV

        # Генерация изображений и сохранение
        for i, (image, text) in enumerate(generator):  # Генератор возвращает картинку и текст
            if isinstance(image, np.ndarray):  # Если возвращён NumPy массив
                pil_image = Image.fromarray(image)  # Конвертируем в PIL Image
            else:
                pil_image = image  # Используем напрямую, если это PIL Image

            # Имя файла изображения
            image_filename = f"image_{i + 1:04d}.jpg"
            image_path = output_dir / image_filename

            # Сохраняем изображение
            pil_image.save(image_path)

            # Сохраняем метку в CSV
            csv_writer.writerow([image_filename, text])
            print(f"Сохранено: {image_path} с меткой '{text}'")

    print(f"\nМетки сохранены в файл: {labels_path}")
