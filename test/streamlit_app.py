import streamlit as st
import pandas as pd
import numpy as np
from model import model, text_preprocess, cv


def load_data():
    uploaded_file = st.file_uploader(label="Выберите данные")
    if uploaded_file is not None:
        # data = uploaded_file.getvalue()
        data = pd.read_csv(uploaded_file, index_col="id")
        return data
    else:
        return None


st.title("Классификафия сообщения о черезвычайном событии")

data = load_data()
st.dataframe(
    data,
    use_container_width=True,
    column_config={"text": st.column_config.Column("text", width="large")},
)

result = st.button("Предсказать событие")
# data.drop(['keyword', 'location'], axis=1)
# df["text"] = df["text"].apply(text_preprocess)

if result:
    # data = pd.read_csv("test_nlp.csv", index_col='id')
    data.drop(["keyword", "location"], axis=1)
    X = data["text"].apply(text_preprocess)
    X = cv.transform(X).toarray()
    preds = model.predict(X)

    st.write("**Результаты предсказания**")
    data["result"] = preds

    st.dataframe(
        data,
        use_container_width=True,
        column_config={"text": st.column_config.Column("text", width="large")},
    )

