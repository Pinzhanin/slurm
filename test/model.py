import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

import string
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

data = pd.read_csv("train_nlp.csv", index_col="id")


# Инициализация целевой переменой
target = data["target"]

# Наивно предположим что место собыития и ключевое слово не влияют на целевую переменную.
df = data.drop(["keyword", "location"], axis=1)

# Удалим дубликаты
df.drop_duplicates(inplace=True, keep="first")


def text_preprocess(text):
    # Преобразование текста в нижний регистр
    text = text.lower()

    # Токенизация текста
    tokens = nltk.word_tokenize(text)

    alphanumeric_tokens = [word for word in tokens if word.isalnum()]

    # Фильтрация стоп-слов и пунктуации
    filtered_tokens = [
        word
        for word in alphanumeric_tokens
        if word not in stopwords.words("english") and word not in string.punctuation
    ]

    # Применение лемматизации
    lemmatized_tokens = [
        WordNetLemmatizer().lemmatize(word) for word in filtered_tokens
    ]

    # Возвращаем результат в виде строки
    return " ".join(lemmatized_tokens)


# Заменим текстовые данные на обработанные
df["text"] = df["text"].apply(text_preprocess)

cv = CountVectorizer()
X = cv.fit_transform(df["text"]).toarray()
y = df["target"].values

model = BernoulliNB()
model.fit(X, y)
