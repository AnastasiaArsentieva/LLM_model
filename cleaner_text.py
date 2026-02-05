import re
from datasets import load_dataset


def is_clean(text):
    # Убираем строки с HTML, ссылками и системным мусором
    if re.search(r'<[^>]+>|http[s]?://\S+|&nbsp;|mso-style|{float:|\[img\]', text):
        return False

    # ПРОВЕРКА НА ДАТУ И ВРЕМЯ (типа 12.05.2023, 15:40:01, 2024-01-01)
    if re.search(r'(\d{2}[.\/-]\d{2}[.\/-]\d{2,4})|(\d{2}:\d{2}(:\d{2})?)', text):
        return False

    # ПРОВЕРКА НА ИМЕНА ПОЛЬЗОВАТЕЛЕЙ (типа "User:", "Иван:", "Admin123:")
    # Если строка начинается как типичный лог чата: "Имя: текст"
    if re.search(r'^[A-Za-zА-Яа-я0-9_.\s]{3,20}:', text):
        return False

    if len(text) < 2 or len(text) > 500:
        return False
    return True


def clean_text_content(text):
    """Дополнительная очистка текста от остаточного мусора внутри строк"""
    # Удаляем конструкции вида [ID12345|Имя] или @username
    text = re.sub(r'\[id\d+\|[^\]]+\]|@\w+', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("--- Загрузка и фильтрация данных ---")
raw_dataset = load_dataset("russian_dialogues", split="train", streaming=True)

clean_data = []
count = 0
target_size = 100000

for item in raw_dataset:
    sample = item['sample']
    if isinstance(sample, list) and len(sample) >= 2:
        context_text = ""
        is_valid_dialogue = True

        for idx, line in enumerate(sample[:4]):
            line_str = str(line)

            # Если хоть одна реплика грязная — бракуем весь диалог
            if not is_clean(line_str):
                is_valid_dialogue = False
                break

            cleaned_line = clean_text_content(line_str)
            label = "Вопрос: " if idx % 2 == 0 else " Ответ: "
            context_text += label + cleaned_line

        if is_valid_dialogue:
            clean_data.append(context_text + " <|endoftext|>")
            count += 1
            if count % 5000 == 0:
                print(f"Собрано {count} чистых диалогов...")

    if count >= target_size:
        break

with open("clean_dialogues.txt", "w", encoding="utf-8") as f:
    for line in clean_data:
        f.write(line + "\n")

print(f"--- Чистый датасет сохранен: {len(clean_data)} строк ---")