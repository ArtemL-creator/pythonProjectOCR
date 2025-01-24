import glob

if __name__ == "__main__":
    # Путь к папке с файлами
    input_dir = "resources/data/train_images_for_ts"

    # Список всех файлов .box
    files = glob.glob(input_dir + "/*.box")

    # Выходной файл для объединения всех символов
    output_box_file = "resources/data/tessdata/all_symbols.box"

    # Открываем файл для записи
    with open(output_box_file, 'w', encoding='utf-8') as output_file:
        # Проходим по всем файлам .box
        for file in files:
            with open(file, 'r', encoding='utf-8') as input_file:
                # Читаем содержимое каждого .box и записываем в общий файл
                output_file.write(input_file.read())

    print(f"Все символы объединены в {output_box_file}")
