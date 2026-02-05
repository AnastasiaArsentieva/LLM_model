import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import torch.nn.functional as F

# --- 1. ПОДГОТОВКА ДАННЫХ И ТОКЕНИЗАТОРА ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 20
VOCAB_SIZE = 10000

print("Загрузка данных...")
df = pd.read_csv("train_sample_500k.csv").astype(str)

# Простейший токенизатор на словах
all_text = " ".join(df['question'].tolist() + df['answer'].tolist()).lower().split()
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
common_words = [word for word, count in Counter(all_text).most_common(VOCAB_SIZE - 4)]
for i, word in enumerate(common_words):
    vocab[word] = i + 4
rev_vocab = {v: k for k, v in vocab.items()}


def encode(text):
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in text.lower().split()[:MAX_LEN]]
    return tokens + [0] * (MAX_LEN - len(tokens))


class ChatDataset(Dataset):
    def __init__(self, df):
        self.q = [torch.tensor(encode(t)) for t in df['question']]
        self.a = [torch.tensor(encode(t)) for t in df['answer']]

    def __len__(self): return len(self.q)

    def __getitem__(self, idx): return self.q[idx], self.a[idx]


dataset = ChatDataset(df)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# --- 2. АРХИТЕКТУРА МОДЕЛИ ---
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 128)
        self.encoder = nn.LSTM(128, 256, batch_first=True)
        self.decoder = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, VOCAB_SIZE)

    def forward(self, x, y):
        _, (h, c) = self.encoder(self.embedding(x))
        out, _ = self.decoder(self.embedding(y), (h, c))
        return self.fc(out)


model = Seq2Seq().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- 3. ЦИКЛ ОБУЧЕНИЯ ---
print(f"Начинаю обучение на {device}...")
model.train()
for epoch in range(1, 11):  # 10 эпох
    total_loss = 0
    for q, a in loader:
        q, a = q.to(device), a.to(device)
        optimizer.zero_grad()
        # Вход декодера (сдвиг для обучения)
        output = model(q, a)
        loss = criterion(output.view(-1, VOCAB_SIZE), a.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Эпоха {epoch}/10, Потери: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "pytorch_bot.pth")
print("Модель сохранена!")

# --- 4. ТЕСТОВЫЙ ОТВЕТ ---
model.eval()
print("\nБот (PyTorch) готов! (Напиши что-нибудь)")
while True:
    input_text = input("Вы: ")
    with torch.no_grad():
        x = torch.tensor([encode(input_text)]).to(device)
        _, (h, c) = model.encoder(model.embedding(x))

        curr_token = torch.tensor([[vocab["<SOS>"]]]).to(device)
        res = []

        for _ in range(MAX_LEN):
            out, (h, c) = model.decoder(model.embedding(curr_token), (h, c))

            # Добавляем "Температуру" (0.7 делает бота более случайным и живым)
            probs = F.softmax(out[0, -1, :] / 0.7, dim=-1)
            token = torch.multinomial(probs, 1).item()  # Случайный выбор на основе вероятности

            if token == vocab["<EOS>"] or token == vocab["<PAD>"]:
                break

            res.append(rev_vocab.get(token, "???"))
            curr_token = torch.tensor([[token]]).to(device)

        print("Бот:", " ".join(res))
    if input_text.lower() == "выход": break

    with torch.no_grad():
        x = torch.tensor([encode(input_text)]).to(device)
        _, (h, c) = model.encoder(model.embedding(x))

        # Генерация (жадный поиск)
        curr_token = torch.tensor([[vocab["<SOS>"]]]).to(device)
        res = []
        for _ in range(MAX_LEN):
            out, (h, c) = model.decoder(model.embedding(curr_token), (h, c))
            token = out.argmax(2).item()
            if token == 0: break
            res.append(rev_vocab.get(token, ""))
            curr_token = torch.tensor([[token]]).to(device)

        print("Бот:", " ".join(res))