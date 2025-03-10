import os
from pathlib import Path

import cv2
import pytesseract
from matplotlib import pyplot as plt


def show_img(image, pre_img):
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Предобработанное изображение")
    plt.imshow(pre_img, cmap="gray")
    plt.axis("off")

    plt.show()


def auto_rotate_image(image):
    """
    Автоматическое определение ориентации текста и исправление поворота.
    Использует функцию orientation в Tesseract.
    """
    # Анализ ориентации текста
    osd = pytesseract.image_to_osd(image)
    rotation_angle = int(osd.split("Rotate:")[1].split("\n")[0])

    # Поворот изображения на основе угла
    if rotation_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        return rotated_image, rotation_angle
    else:
        return image, 0


def preprocess_image(image_path, output_path=None):
    # Загрузка изображения
    print(f"Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Файл не найден или путь указан неверно: {image_path}")

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Устранение шума и сглаживание
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Улучшение контрастности с помощью CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Опциональный автоматический поворот на основе ориентации текста
    try:
        auto_rotated, _ = auto_rotate_image(enhanced)
    except Exception as e:
        # print(f"Ошибка автоматического поворота: {e}")
        auto_rotated = enhanced

    # Сохранение изображения, если указан output_path
    if output_path:
        cv2.imwrite(str(output_path), enhanced)

    # # Отображение результата
    show_img(image, enhanced)

    return enhanced


def recognize_text(image):
    """
    Распознавание текста с помощью Tesseract.
    """
    # Обычное разбиение на блоки текста (по умолчанию --psm 3)
    # Устанавливаем переменную окружения для Tesseract
    # os.environ['TESSDATA_PREFIX'] = r'tesseract/tessdata'
    os.environ['TESSDATA_PREFIX'] = r'tesstrain/data'

    # Убедимся, что переменная окружения установлена
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    if tessdata_prefix is None:
        print("Переменная окружения TESSDATA_PREFIX не установлена.")
        return

    print(f"Переменная окружения TESSDATA_PREFIX установлена: {tessdata_prefix}")

    text = pytesseract.image_to_string(image, lang='my_model', config='--psm 6')
    return text


if __name__ == "__main__":
    input_image = str(Path('resources', 'input', '13_388_copy.jpg'))
    output_image = str(Path('resources', 'output', '13_388_copy.jpg'))
    # Путь к tesseract

    preprocessed_image = preprocess_image(input_image, output_image)
    cv2.imwrite(str(output_image), preprocessed_image)

    recognized_text = recognize_text(preprocessed_image)

    # Распознавание текста
    print("Распознанный текст:")
    print(recognized_text)

    output_text = Path('resources', 'output', 'text.txt')
    # Сохранение текста в файл
    with open('data/recognized_text.txt', "w", encoding="utf-8") as text_file:
        text_file.write(recognized_text)
