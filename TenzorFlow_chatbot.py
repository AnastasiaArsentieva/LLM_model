import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. НАСТРОЙКИ (Синхронизировали VOCAB_SIZE) ---
MAX_LEN = 20
VOCAB_SIZE = 15000  # Должно быть равно или больше num_words в токенизаторе
LATENT_DIM = 256
EMB_DIM = 128

# --- 2. ПОДГОТОВКА ДАННЫХ ---
print("Загрузка и подготовка данных...")
# Загружаем 500к, но для обучения перемешиваем и берем часть, если нужно
df = pd.read_csv("train_sample_500k.csv").astype(str)
df['answer'] = df['answer'].apply(lambda x: f"startseq {x} endseq")

# Важно: обучаем токенизатор на ВСЕХ данных, но ограничиваем его VOCAB_SIZE
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token="<UNK>")
tokenizer.fit_on_texts(df['question'] + df['answer'])


def prepare_sequences(texts, max_len=MAX_LEN):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding='post')


# Используем выборку в 500к для обучения, чтобы ускорить процесс
df_train = df.sample(500000, random_state=42)
X = prepare_sequences(df_train['question'])
Y = prepare_sequences(df_train['answer'])

# --- 3. АРХИТЕКТУРА МОДЕЛИ ---
# Энкодер
enc_in = Input(shape=(MAX_LEN,))
enc_emb_layer = Embedding(VOCAB_SIZE, EMB_DIM)
enc_lstm_layer = LSTM(LATENT_DIM, return_state=True)
enc_out, state_h, state_c = enc_lstm_layer(enc_emb_layer(enc_in))
enc_states = [state_h, state_c]

# Декодер
dec_in = Input(shape=(MAX_LEN,))
dec_emb_layer = Embedding(VOCAB_SIZE, EMB_DIM)
dec_lstm_layer = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
dec_out_layer = Dense(VOCAB_SIZE, activation='softmax')

# Связка для обучения
d_lstm_out, _, _ = dec_lstm_layer(dec_emb_layer(dec_in), initial_state=enc_states)
d_final_out = dec_out_layer(d_lstm_out)

model = Model([enc_in, dec_in], d_final_out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# --- 4. ОБУЧЕНИЕ ---
print(f"Начинаю обучение на {len(X)} строках...")
model.fit(
    [X, Y],
    np.expand_dims(Y, -1),
    batch_size=64,
    epochs=10
)

model.save("my_chatbot_model.keras")  # Используем новый формат
print("Модель сохранена!")


# --- 5. ФУНКЦИЯ ДЛЯ ОБЩЕНИЯ ---
def chat():
    print("\nБот готов! Пиши (или 'выход' для стопа):")

    # Модели для генерации
    encoder_model = Model(enc_in, enc_states)

    dec_state_h = Input(shape=(LATENT_DIM,))
    dec_state_c = Input(shape=(LATENT_DIM,))
    dec_states_in = [dec_state_h, dec_state_c]

    # Одиночный вход для генерации по словам
    dec_single_in = Input(shape=(1,))
    d_emb_s = dec_emb_layer(dec_single_in)
    d_lstm_s, h_s, c_s = dec_lstm_layer(d_emb_s, initial_state=dec_states_in)
    d_out_s = dec_out_layer(d_lstm_s)

    decoder_model = Model([dec_single_in] + dec_states_in, [d_out_s, h_s, c_s])

    reverse_word_map = {v: k for k, v in tokenizer.word_index.items()}
    start_index = tokenizer.word_index.get('startseq', 1)
    end_index = tokenizer.word_index.get('endseq', 2)

    while True:
        text = input("Вы: ")
        if text.lower() == 'выход': break

        seq = prepare_sequences([text])
        states_value = encoder_model.predict(seq, verbose=0)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = start_index

        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

            # Добавим немного случайности (sampling), чтобы бот не зацикливался
            probs = output_tokens[0, -1, :]
            # Чем меньше 1.0, тем бот строже; чем больше 1.0, тем бот "креативнее"
            sampled_token_index = np.random.choice(range(VOCAB_SIZE), p=probs / np.sum(probs))

            sampled_word = reverse_word_map.get(sampled_token_index, '')

            if sampled_word == 'endseq' or len(decoded_sentence) > MAX_LEN:
                stop_condition = True
            elif sampled_word != 'startseq' and sampled_word != '<UNK>':
                decoded_sentence.append(sampled_word)

            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        print(f"Бот: {' '.join(decoded_sentence)}")


chat()