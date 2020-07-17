from __future__ import annotations

from enum import Enum
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from disaster_tweets.vocab import Vocabulary
from disaster_tweets.tokenizer import tokenize


class SplitType(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class TweetVectorizer(object):
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    @classmethod
    def from_dataframe(cls, tweet_df: pd.DataFrame):
        tweet_vocab = Vocabulary(add_unknown=True)

        for tweet in tweet_df.text:
            for word in tokenize(tweet):
                tweet_vocab.add(word)

        return cls(tweet_vocab)

    def vectorize(self, tweet: str) -> np.array:
        bow = np.zeros(len(self.vocab), dtype=np.float32)

        for token in tokenize(tweet):
            bow[self.vocab.lookup(token)] = 1

        return bow


class DisasterTweetDataset(Dataset):
    def __init__(self, tweet_df: pd.DataFrame, tweet_vectorizer: TweetVectorizer) -> None:
        self.tweet_df = tweet_df
        self.tweet_vectorizer = tweet_vectorizer

        train_df, test_df = train_test_split(tweet_df)
        self.dataset = {
            SplitType.TRAIN: train_df,
            SplitType.VAL: test_df,
        }
        self.target = SplitType.TRAIN

    @classmethod
    def from_csv(cls, csv_file: str) -> DisasterTweetDataset:
        dataset = pd.read_csv(csv_file)
        dataset.fillna("", inplace=True)
        return cls(dataset, TweetVectorizer.from_dataframe(dataset))

    def set_split(self, split: SplitType) -> None:
        self.target = split

    def get_num_batches(self, batch_size: int) -> int:
        return len(self) // batch_size

    def __len__(self) -> int:
        return len(self.dataset[self.target])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataset[self.target].iloc[index]
        x_data = self.tweet_vectorizer.vectorize(row.text)
        y_target = float(row.target)

        return {"x_data": x_data, "y_target": y_target}


def generate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    device: torch.device = torch.device("cpu"),
):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)

        yield out_data_dict
