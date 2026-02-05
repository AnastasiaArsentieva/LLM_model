import os
import torch
import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. НАСТРОЙКИ ПУТЕЙ ---
# Указываем путь к папке, где лежат чекпоинты
BASE_MODEL_DIR = "./rugpt3_chatbot/checkpoint-2000"


def get_latest_checkpoint(base_path):
    try:
        checkpoints = [d for d in os.listdir(base_path) if d.startswith("checkpoint-")]
        if not checkpoints: return base_path
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        return os.path.join(base_path, latest)
    except Exception:
        return base_path


MODEL_PATH = get_latest_checkpoint(BASE_MODEL_DIR)
API_TOKEN = 'ВСТАВЬ СВОЙ ТОКЕН ТУТ'  # ВСТАВЬ СВОЙ ТОКЕН ТУТ

# --- 2. ЗАГРУЗКА МОДЕЛИ ---
logging.basicConfig(level=logging.INFO)
print(f"Загрузка модели из: {MODEL_PATH}...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 3. ЛОГИКА БОТА ---
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


def generate_answer(user_text):
    # Форматируем как при обучении
    prompt = f"Вопрос: {user_text} Ответ:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,  # Длина ответа
            do_sample=True,  # ВКЛЮЧАЕМ вариативность (обязательно!)
            temperature=0.6,  # Баланс (0.1 - скучно, 1.0 - бред)
            top_p=0.9,  # Выбор самых вероятных слов
            repetition_penalty=1.2,  # Штраф за повторы (не ставь 20!)
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # Очищаем ответ от промпта
    if "Ответ:" in decoded:
        answer = decoded.split("Ответ:")[-1].strip()
    else:
        answer = decoded.replace(prompt, "").strip()
    return answer


# Команда /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Бот запущен и готов к общению!")


# Ответ на любое сообщение
@dp.message(F.text)
async def chat(message: types.Message):
    # Показываем статус "печатает"
    await bot.send_chat_action(message.chat.id, action="typing")

    try:
        loop = asyncio.get_event_loop()
        # Запускаем тяжелую генерацию в отдельном потоке, чтобы бот не "виснул"
        reply = await loop.run_in_executor(None, generate_answer, message.text)

        if not reply:
            reply = "Я задумался и не нашел слов..."

        await message.reply(reply)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await message.reply("Произошла ошибка при генерации.")


async def main():
    print("Бот вышел в онлайн!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())