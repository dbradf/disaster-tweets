from enum import Enum
import random
from time import perf_counter

import pandas as pd
import click
from sklearn.model_selection import train_test_split
import spacy
from spacy.util import minibatch
import torch
from tqdm import tqdm

is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    print("gpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


class Labels(Enum):

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@click.command()
@click.option("--n-epochs", default=10)
@click.option("--dataset-file", required=True)
@click.option("--batch-size", default=8, type=int)
def main(n_epochs: int, dataset_file: str, batch_size: int):
    nlp = spacy.load("en_trf_bertbaseuncased_lg")
    textcat = nlp.create_pipe("trf_textcat", config={"exclusive_classes": True})

    possible_labels = [l.value for l in Labels]

    for label in possible_labels:
        textcat.add_label(label)
    nlp.add_pipe(textcat)

    dataset = pd.read_csv(dataset_file)
    dataset.fillna("", inplace=True)

    train, test = train_test_split(dataset, shuffle=True)

    train_prepared_data = []
    for _, x in train.iterrows():
        train_prepared_data.append((x.text, {'cats': {
            Labels.POSITIVE.value: 1.0 if x.target == 1 else 0.0,
            Labels.NEGATIVE.value: 1.0 if x.target != 1 else 0.0,
        }}))

    optimizer = nlp.resume_training()
    for i in tqdm(range(n_epochs)):
        losses = {}
        random.shuffle(train_prepared_data)
        start = perf_counter()
        for i, batch in enumerate(minibatch(train_prepared_data, batch_size)):
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            # print(f"{i} / {len(train_prepared_data)/batch_size}")

        end = perf_counter()
        print(f"{i}: loss: {losses}, time: {end - start}")

    nlp.to_disk("bert-textcat")

    def predict(cats):
        if cats[Labels.POSITIVE.value] >= 0.5:
            return 1
        return 0

    correct = 0
    count = 0
    for _, doc in test.iterrows():
        count += 1
        res = nlp(doc.text)
        print(f'{res}: {res.cats}')
        if doc.target == predict(res.cats):
            correct += 1

    print(f"test accuracy: {correct} / {count} == {correct / count}")
