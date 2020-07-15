from datetime import datetime
import pickle
from typing import Optional

import click
import pandas as pd
import spacy

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from disaster_tweets.tokenizer import tokenize

TARGET = "target"


def create_pipeline(
    classifier: BaseEstimator, n_grams: int = 1, reduction: Optional[int] = None
) -> Pipeline:
    steps = [
        (
            "column",
            ColumnTransformer(
                [
                    (
                        "text",
                        TfidfVectorizer(
                            tokenizer=tokenize,
                            preprocessor=None,
                            lowercase=False,
                            ngram_range=(1, n_grams),
                        ),
                        "text",
                    ),
                    ("keyword", OneHotEncoder(handle_unknown="ignore"), ["keyword"]),
                    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
                ],
                remainder="drop",
                transformer_weights={"text": 1, "keyword": 0, "location": 0},
            ),
        ),
    ]

    if reduction is not None:
        steps.append(("reduction", TruncatedSVD(n_components=reduction)))

    steps.append(("classifier", classifier))

    return Pipeline(steps)


@click.group()
@click.option("--dataset-file", required=True)
@click.pass_context
def cli(ctx, dataset_file):
    dataset = pd.read_csv(dataset_file)
    dataset.fillna("", inplace=True)
    ctx.ensure_object(dict)
    ctx.obj["dataset"] = dataset


def train_model(name, clf, train, test, **kwargs):
    print(f"{name}...")
    start = datetime.now()
    model = create_pipeline(clf, **kwargs)
    model.fit(train, train[TARGET])
    end = datetime.now()

    y_pred = model.predict(test)
    print(f"Training time: {(end - start).total_seconds()}s")
    print(classification_report(test[TARGET], y_pred))

    return model


@cli.command()
@click.option("--train-percent", type=float, default=0.8)
@click.option("--n-grams", type=int, default=1)
@click.option("--save")
@click.pass_context
def train(ctx, train_percent, save, n_grams):
    dataset: pd.DataFrame = ctx.obj["dataset"]

    train, test = train_test_split(dataset, train_size=train_percent, shuffle=True)

    models = {
        "SGD": SGDClassifier(n_jobs=-1),
        "SGD Log Loss": SGDClassifier(loss="log", n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=500, n_jobs=-1),
        "SVM": SVC(),
        "Ridge": RidgeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    {name: train_model(name, clf, train, test, n_grams=n_grams) for name, clf in models.items()}

    voter_models = [
        ("SGD", SGDClassifier(n_jobs=-10)),
        ("SGD Log Loss", SGDClassifier(loss="log", n_jobs=-10)),
        ("Random Forest", RandomForestClassifier(n_estimators=500, n_jobs=-1)),
        ("SVM", SVC()),
        ("Ridge", RidgeClassifier()),
        # ("GradientBoosting", GradientBoostingClassifier()),
    ]
    voter = VotingClassifier(voter_models, n_jobs=-1)

    voter_model = train_model("Voter", voter, train, test, n_grams=n_grams)

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
