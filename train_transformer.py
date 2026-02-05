import pandas as pd
import torch
import os
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. Настройки окружения
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

# Проверка доступности CUDA для настроек
IS_CUDA = torch.cuda.is_available()
print(f"--- Доступность CUDA: {IS_CUDA} ---")

# 2. Конфигурация
MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
INPUT_FILE = "train_sample_500k.csv"
OUTPUT_DIR = "./rugpt3_chatbot"

print(f"--- Шаг 1: Загрузка данных из {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE):
    print(f"Ошибка: Файл {INPUT_FILE} не найден!")
    exit()

df = pd.read_csv(INPUT_FILE).astype(str)

# 3. Подготовка токенизатора и модели
print("--- Шаг 2: Загрузка модели и токенизатора ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель. Веса теперь подтянутся корректно.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# 4. Форматирование и токенизация
def format_and_tokenize(examples):
    texts = [
        f"Вопрос: {q} Ответ: {a}{tokenizer.eos_token}"
        for q, a in zip(examples['question'], examples['answer'])
    ]
    return tokenizer(texts, truncation=True, max_length=128, padding="max_length")


print("--- Шаг 3: Токенизация датасета ---")
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

# 5. Настройка параметров обучения (УСКОРЕННЫЙ РЕЖИМ)
print("--- Шаг 4: Настройка параметров обучения ---")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,

    # ПАРАМЕТРЫ СКОРОСТИ
    per_device_train_batch_size=8 if IS_CUDA else 2,  # Если нет GPU, ставим батч поменьше
    gradient_accumulation_steps=4 if IS_CUDA else 16,  # Компенсируем размер батча

    learning_rate=2e-5,
    weight_decay=0.01,

    # СОХРАНЕНИЕ И ЛОГИ
    save_steps=500,  # Промежуточный чекпоинт каждые 500 шагов
    logging_steps=10,  # Видеть прогресс каждые 10 шагов
    save_total_limit=5,  # Хранить только 5 последних чекпоинтов

    # ОПТИМИЗАЦИЯ ПОД КАРТУ (Безопасные настройки)
    fp16=IS_CUDA,  # Смешанная точность включится только на GPU
    # Ускорение для новых карт NVIDIA (RTX 30/40), если CUDA доступна
    tf32=IS_CUDA and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,

    group_by_length=True,  # Ускоряет обучение, группируя похожие фразы
    warmup_steps=500,
    prediction_loss_only=True,
    report_to="none"
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 6. Запуск
print("--- Шаг 5: ЗАПУСК ОБУЧЕНИЯ ---")
try:
    trainer.train()
except Exception as e:
    print(f"Произошла ошибка при обучении: {e}")

# Финальное сохранение
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"--- ОБУЧЕНИЕ ЗАВЕРШЕНО! ---")
print(f"Модель и токенизатор сохранены в: {OUTPUT_DIR}")