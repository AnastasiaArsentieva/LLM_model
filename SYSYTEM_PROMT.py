import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = r"путь к последнему чекпоиинту"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_HISTORY = 4  # Глубина памяти

print(f"--- Загрузка релизной модели на {DEVICE} ---")
tokenizer = AutoTokenizer.from_pretrained("путь к обученой модели")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Системная установка (задает вектор ИИ-помощника)
SYSTEM_PROMPT = [
    "Вопрос: Кто ты?",
    "Ответ: Я твой виртуальный помощник, искусственный интеллект. Я здесь, чтобы помогать тебе и общаться. <|endoftext|>"
]
chat_history = list(SYSTEM_PROMPT)


def clean_bot_reply(text):
    """Финальная чистка от HTML, технических символов и форумного мусора"""
    # 1. Удаляем HTML-теги и сущности
    text = re.sub(r'<[^>]*>|&nbsp;|&quot;|&gt;|&lt;', '', text)

    # 2. Удаляем технические атрибуты (стили, классы и т.д.)
    bad_patterns = [
        r'\b(span|div|class|font|color|style|padding|align|table|border)\b',
        r'http\S+|www\S+',
        r'[\|><\^\-_~\\/\[\]]'
    ]
    for pattern in bad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Отсекаем, если модель начала генерировать за пользователя
    text = text.split("Вопрос:")[0].split("Ответ:")[0]

    # 4. Схлопываем лишние пробелы и знаки
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Убираем дубли знаков (???, !!!)

    return text.strip().capitalize()


def get_response(user_text):
    chat_history.append(f"Вопрос: {user_text}")

    # Склеиваем контекст
    context = "\n".join(chat_history[-MAX_HISTORY:])
    prompt = f"{context}\n Ответ:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,  # Даем запас для длинных мыслей
            do_sample=True,
            temperature=0.4,  # Сдержанность
            top_p=0.8,  # Точность
            repetition_penalty=0.8,  # Жесткий запрет на повторы (никаких Москва-Москва)
            no_repeat_ngram_size=3,  # Уникальность фраз
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Извлекаем ответ после последней метки "Ответ:"
    parts = full_text.split("Ответ:")
    raw_answer = parts[-1] if len(parts) > 1 else ""

    clean_answer = clean_bot_reply(raw_answer)

    if not clean_answer or len(clean_answer) < 2:
        clean_answer = "Я задумался... Можешь уточнить вопрос?"

    chat_history.append(f"Ответ: {clean_answer}")
    return clean_answer


# --- ЦИКЛ ДИАЛОГА ---
print("\n" + "=" * 20)
print("РЕЛИЗНАЯ ВЕРСИЯ БОТА ГОТОВА")
print("Напиши 'выход' для завершения")
print("=" * 30 + "\n")

while True:
    try:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ['выход', 'stop', 'exit']:
            print("Бот: До встречи!")
            break
        if not user_input:
            continue

        response = get_response(user_input)
        print(f"Бот: {response}")
        print("-" * 10)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\n[Ошибка]: {e}")