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


def preprocess_query(query: str) -> list[str]:
    if query is None or pd.isna(query):
        return []

    query = query.lower()
    query = re.sub(f"[{string.punctuation}]", "", query)
    tokens = simple_preprocess(
        query, deacc=True
    )  # deacc=True removes accents and punctuations
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    tokens = ["[S]"] + tokens + ["[E]"]
    return tokens
