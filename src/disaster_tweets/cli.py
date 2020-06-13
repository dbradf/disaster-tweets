from datetime import datetime
import pickle

import click
import pandas as pd
import spacy

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


TARGET = "target"

nlp = spacy.load('en')


def tokenize(text):
    tokens = [token for token in nlp(text) if not token.is_stop]
    for token in tokens:
        yield token.norm_
    for token, token2 in zip(tokens, tokens[1:]):
        yield f"{token.norm_} {token2.norm_}"
    # return [token.norm_ for token in tokens if not token.is_stop]


def create_pipeline(classifier: BaseEstimator) -> Pipeline:
    steps = [
        (
            "column",
            ColumnTransformer(
                [
                    (
                        "text",
                        TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False),
                        "text",
                    ),
                    ("keyword", OneHotEncoder(handle_unknown="ignore"), ["keyword"]),
                    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
                ],
                remainder="drop",
            ),
        ),
        ("classifier", classifier),
    ]

    return Pipeline(steps)


@click.group()
@click.option("--dataset-file", required=True)
@click.pass_context
def cli(ctx, dataset_file):
    dataset = pd.read_csv(dataset_file)
    dataset.fillna("", inplace=True)
    ctx.ensure_object(dict)
    ctx.obj["dataset"] = dataset


def train_model(name, clf, train, test):
    print(f"{name}...")
    start = datetime.now()
    model = create_pipeline(clf)
    model.fit(train, train[TARGET])
    end = datetime.now()

    y_pred = model.predict(test)
    print(f"Training time: {(end - start).total_seconds()}s")
    print(classification_report(test[TARGET], y_pred))

    return model


@cli.command()
@click.option("--train-percent", type=float, default=0.8)
@click.option("--save")
@click.pass_context
def train(ctx, train_percent, save):
    dataset: pd.DataFrame = ctx.obj["dataset"]

    train, test = train_test_split(dataset, train_size=train_percent, shuffle=True)

    models = {
        "SGD": SGDClassifier(n_jobs=10),
        "SGD Log Loss": SGDClassifier(loss="log", n_jobs=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "SVM": SVC(),
        "Ridge": RidgeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
    }
    trained_models = {name: train_model(name, clf, train, test) for name, clf in models.items()}

    voter_models = [
        ("SGD", SGDClassifier(n_jobs=10)),
        ("SGD Log Loss", SGDClassifier(loss="log", n_jobs=10)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
        ("SVM", SVC()),
        ("Ridge", RidgeClassifier()),
        # ("GradientBoosting", GradientBoostingClassifier()),
    ]
    voter = VotingClassifier(voter_models, n_jobs=-1)
    
    voter_model = train_model("Voter", voter, train, test)

    if save:
        with open(save, "wb") as f:
            pickle.dump(voter_model, f)


@cli.command()
@click.option("--model-file", required=True)
@click.option("--output-file", required=True)
@click.pass_context
def test(ctx, model_file, output_file):
    dataset: pd.DataFrame = ctx.obj["dataset"]

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    dataset["target"] = ""

    dataset["target"] = model.predict(dataset)

    print(dataset.describe())

    # output = dataset.loc("id", "target")
    dataset.to_csv(output_file, columns=["id", "target"], index=False)


def main():
    """Entry point into commandline."""
    return cli(obj={})


if __name__ == "__main__":
    main()
