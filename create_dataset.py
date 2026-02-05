import pandas as pd
from datasets import load_dataset

print("Загрузка данных...")
dataset = load_dataset("russian_dialogues" split="train")

# Превращаем в список для быстрой обработки
raw_data = dataset['sample']
questions = []
answers = []

print("Распаковка...")
for item in raw_data:
    # Если это список и в нем 2 и более элементов
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        # Берем первый элемент как вопрос, второй как ответ
        questions.append(str(item[0]))
        answers.append(str(item[1]))

# Создаем DataFrame
df = pd.DataFrame({'question': questions, 'answer': answers})

# Минимальная чистка (только совсем пустые)
df = df.dropna()
df = df[df['question'].str.strip() != ""]
df = df[df['answer'].str.strip() != ""]

print(f"Теперь {len(df)} пар данных.")
df.to_csv("cleaned_dialogues.csv", index=False, encoding='utf-8')

