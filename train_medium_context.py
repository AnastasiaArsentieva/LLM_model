import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. Конфигурация
MODEL_NAME = "путь к базовой модели" 
OUTPUT_DIR = "./rugpt3_medium_100k_context" - создание директории для хранения

print("--- Шаг 1: Загрузка 100 000 строк (многоходовые диалоги) ---")
raw_dataset = load_dataset("russian_dialogues", split="train", streaming=True) - файл с диалогами

data_list = []
# Собираем 100 000 примеров
for i, item in enumerate(raw_dataset):
    if i >= 100000: break
    sample = item['sample']

    # Пытаемся собрать длинную цепочку из примера, если там больше 2 реплик
    if isinstance(sample, list) and len(sample) >= 2:
        # Формируем цепочку: Вопрос: ... Ответ: ... Вопрос: ... Ответ: ...
        context_text = ""
        for idx, line in enumerate(sample[:4]):  # Берем до 4 реплик для памяти
            label = "Вопрос: " if idx % 2 == 0 else " Ответ: "
            context_text += label + str(line)

        data_list.append(context_text + " <|endoftext|>")

print(f"Подготовлено {len(data_list)} строк контекста.")

# 2. Модель и Токенизатор
print("--- Шаг 2: Загрузка MEDIUM модели (займет время) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# Medium модель весит около 1.5 Гб
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# 3. Токенизация
def tokenize_func(examples):
    # Увеличиваем max_length до 128, чтобы влезала история
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")


dataset = Dataset.from_dict({"text": data_list})
tokenized_dataset = dataset.map(tokenize_func, batched=True)

# 4. Настройки обучения (ДЛЯ CPU И MEDIUM)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # 3 эпохи для "закрепления" памяти
    per_device_train_batch_size=1,  # Только 1, так как Medium тяжелая
    gradient_accumulation_steps=16,  # Увеличиваем шаг для стабильности
    learning_rate=2e-5,  # Чуть ниже для крупной модели
    logging_steps=50,
    save_steps=100,  # Сохраняем каждые 100 шагов
    save_total_limit=2,
    use_cpu=True,  # Процессор
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("--- Шаг 3: СТАРТ ДЛИТЕЛЬНОГО ОБУЧЕНИЯ (на несколько дней) ---")
trainer.train()

# Финальное сохранение
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"ПОБЕДА! Medium модель сохранена в {OUTPUT_DIR}")


