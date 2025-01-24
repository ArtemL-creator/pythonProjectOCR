import subprocess
from pathlib import Path


def train_model(train_listfile, eval_listfile, model_output_dir, traineddata, max_iterations=10000):
    """
    Обучение модели с использованием lstmtraining.
    """
    model_output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_model = model_output_dir / "rus_checkpoint.lstm"

    # Этап 1: дообучение модели
    command_1 = [
        "lstmtraining",
        "--model_output", str(intermediate_model),
        "--traineddata", str(traineddata),
        "--train_listfile", str(train_listfile),
        "--eval_listfile", str(eval_listfile),
        "--max_iterations", str(max_iterations),
    ]
    print("Запуск обучения модели...")
    subprocess.run(command_1, check=True)

    final_model = model_output_dir / "rus_final.traineddata"

    # Этап 2: завершение обучения и сохранение финальной модели
    command_2 = [
        "lstmtraining",
        "--stop_training",
        "--continue_from", str(intermediate_model),
        "--traineddata", str(traineddata),
        "--model_output", str(final_model),
    ]
    print("Создание финальной модели...")
    subprocess.run(command_2, check=True)

    print(f"Обучение завершено. Финальная модель сохранена как: {final_model}")


if __name__ == "__main__":
    # Пути к файлам
    traineddata = Path("resources/data/tessdata/rus.lstm")  # Исходная модель
    train_listfile = Path("resources/data/lstmf_txt/training_files.txt")  # Список тренировочных файлов
    eval_listfile = Path("resources/data/lstmf_txt/evaluation_files.txt")  # Список файлов для оценки
    output_dir = Path("resources/data/output")  # Директория для сохранения моделей

    # Обучение модели
    train_model(train_listfile, eval_listfile, output_dir, traineddata, max_iterations=10000)
