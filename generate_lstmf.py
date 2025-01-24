import os
import shutil
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Путь к директории с изображениями и метками
    image_dir = Path('resources/data/train_images_for_ts')
    output_dir = Path('resources/data/lstmf_files')  # Папка для .lstmf файлов
    output_dir.mkdir(parents=True, exist_ok=True)

    # Список изображений в формате .tif
    images = list(image_dir.glob("*.tif"))

    if not images:
        print("Нет изображений для конвертации!")
        exit(1)

    # Проверяем наличие Tesseract
    if not shutil.which("tesseract"):
        print("Tesseract не установлен или не доступен в PATH.")
        exit(1)

    # Автоматическая конвертация в .lstmf
    for image in images:
        box_file = image.with_suffix(".box")  # Проверяем наличие .box файла
        if not box_file.exists():
            print(f"Пропущено: {image}, так как отсутствует {box_file}")
            continue

        # Команда для tesseract: конвертация в .lstmf
        output_lstmf = output_dir / f"{image.stem}.lstmf"
        command = [
            "tesseract",
            str(image),  # Входное изображение
            str(output_lstmf.with_suffix("")),  # Выходной файл без расширения
            "--psm", "6",  # PSM 6: распознавание строки
            "-l", "rus",  # Только русский язык
            "lstm.train"  # Преобразование в формат для обучения
        ]

        # Запускаем процесс
        print(f"Конвертация {image} -> {output_lstmf}")
        subprocess.run(command, check=True)

    print(f"Конвертация завершена. Файлы .lstmf сохранены в {output_dir}")
