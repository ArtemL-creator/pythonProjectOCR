from pathlib import Path

if __name__ == "__main__":
    lstmf_dir = Path('../resources/data/lstmf_files')
    # train_listfile = lstmf_txt / "training_files.txt"
    train_listfile = Path('../resources/data/lstmf_txt/training_files.txt')
    # eval_listfile = lstmf_txt / "evaluation_files.txt"
    eval_listfile = Path('../resources/data/lstmf_txt/evaluation_files.txt')

    # Получаем список всех .lstmf файлов
    lstmf_files = list(lstmf_dir.glob("*.lstmf"))
    if not lstmf_files:
        print("Нет .lstmf файлов для обучения!")
        exit(1)

    # Делим на тренировочные и валидационные данные
    train_split = int(len(lstmf_files) * 0.9)  # 90% на обучение, 10% на проверку
    train_files = lstmf_files[:train_split]
    eval_files = lstmf_files[train_split:]

    # Сохраняем списки
    with open(train_listfile, "w") as train_file:
        train_file.writelines(f"{file}\n" for file in train_files)

    with open(eval_listfile, "w") as eval_file:
        eval_file.writelines(f"{file}\n" for file in eval_files)

    print(f"Списки файлов сохранены: {train_listfile}, {eval_listfile}")
