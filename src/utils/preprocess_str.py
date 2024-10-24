import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess

nltk.download("stopwords")
nltk.download("punkt")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_list(tokens: list[str]) -> list[str]:
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens


def preprocess_str(text: str) -> list[str]:
    if text is None or pd.isna(text):
        return []

    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = simple_preprocess(
        text, deacc=True
    )  # deacc=True removes accents and punctuations

    tokens = preprocess_list(tokens)
    return tokens
