import pandas as pd
from datasets import load_dataset

print("Подключаюсь к репозиторию и ищу актуальные файлы...")

# Загружаем датасет с параметром verification_mode="no_checks"
#  библиотека игнорирует ошибки в метаданных
dataset = load_dataset(
        "russian_dialogues",
        split="train",
        revision="main",
        verification_mode="no_checks"
    )

print("Данные успешно получены!")

# Превращаем в Pandas
df = dataset.to_pandas()

print(f"Загружено строк: {len(df)}")
print(df.head())

# Сохраняем локально
df.to_csv("russian_dialogues.csv", index=False)
print("Файл сохранен как russian_dialogues.csv")



