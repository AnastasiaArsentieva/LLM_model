import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 1. Настройки
CHECKPOINT_PATH = r"путь нахождения чекпоинта"
DATA_PATH = "путь нахождения датасета" # Твой  файл
DEVICE = "cpu"

print(f"--- Расчет Perplexity для: {CHECKPOINT_PATH} ---")

tokenizer = AutoTokenizer.from_pretrained("путь к обученной модели")
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
model.eval()

# 2. Берем небольшую выборку для теста (например, 100 строк)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    test_lines = [f.readline().strip() for _ in range(100)]

nlls = [] # Negative Log-Likelihoods
print("Обработка тестовых фраз...")

for line in test_lines:
    if not line: continue
    encodings = tokenizer(line, return_tensors="pt").to(DEVICE)
    target_ids = encodings.input_ids.clone()

    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

# 3. Считаем итог
avg_nll = torch.stack(nlls).mean()
ppl = math.exp(avg_nll)

print("-" * 30)
print(f"Результат Loss (средний): {avg_nll:.4f}")
print(f"PERPLEXITY (PPL): {ppl:.2f}")
print("-" * 30)

if ppl > 50:
    print("Статус: Модель еще 'глупая', нужно учить дальше.")
elif 20 < ppl <= 50:
    print("Статус: Хороший уровень! Бот уже понимает контекст.")
else:
    print("Статус: Отлично! Бот очень уверен в своих ответах.")
