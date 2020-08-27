import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def tokenize(text):
    return text.split()


def get_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file)
    dataset.fillna("", inplace=True)

    return train_test_split(dataset, train_size=0.8, shuffle=True)
    

def report_to_df(report):
    return pd.DataFrame(report).transpose()[["precision", "recall", "f1-score"]]


def classification_heatmap(report_df, title):
    sns.set()

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(report_df.iloc[[0, 1]], annot=True, fmt=".2%", linewidths=.5, ax=ax, cmap="YlGnBu").set_title(title)


DEFAULT_WEIGHTS = {"text": 1, "keyword": 0, "location": 0}

# Define our pipeline.
def create_pipeline(classifier, vectorizer=CountVectorizer, weights=None, tokenizer=tokenize):
    steps = [
        (
        "column", ColumnTransformer(
            [
                ("text", vectorizer(tokenizer=tokenizer, preprocessor=None, lowercase=False), "text"),
                ("keyword", OneHotEncoder(handle_unknown="ignore"), ["keyword"]),
                ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
            ],
            remainder="drop",
            transformer_weights=weights if weights else DEFAULT_WEIGHTS,
        ))
    ]
    
    steps.append(("classifier", classifier))
    return Pipeline(steps)