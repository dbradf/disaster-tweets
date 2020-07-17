from datetime import datetime

import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from disaster_tweets.tweet_data import DisasterTweetDataset, SplitType


class LinearTweetClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(in_features=embed_dim, out_features=n_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class RunningValue(object):
    def __init__(self, batch_size: int, initial_value: float = 0.0):
        self.batch_size = batch_size
        self.value = initial_value

    def add(self, amt: float):
        self.value += (amt - self.value) / (self.batch_size - 1)


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def generate_batch(batch):
    label = torch.tensor([entry["y_target"] for entry in batch]).long()
    text = [torch.from_numpy(entry['x_data']).long() for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1])
    text = torch.cat(text)
    return text, offsets, label


def trainer(
    dataset: DisasterTweetDataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_func,
    num_epochs: int,
    batch_size: int,
    shuffle: bool = True,
    device: torch.device = torch.device("cpu")
):
    for epoch in tqdm(range(num_epochs)):
        train_acc = 0.0
        train_loss = 0.0
        count = 0

        dataset.set_split(SplitType.TRAIN)
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=generate_batch,
        )

        model.train()
        for text, offsets, target in data_loader:
            optimizer.zero_grad()

            text, offsets, target = text.to(device), offsets.to(device), target.to(device)
            y_pred = model(text, offsets)

            loss = loss_func(y_pred, target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            train_acc += (y_pred.argmax(1) == target).sum().item()
            count += batch_size

        print(f"{datetime.now()}: Epoch {epoch}: Train Acc {train_acc / count}: Train Loss {train_loss / count}")

        dataset.set_split(SplitType.VAL)
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=generate_batch,
        )

        val_loss = 0.0
        val_acc = 0.0
        count = 0.0

        model.eval()
        for text, offsets, target in data_loader:
            text, offsets, target = text.to(device), offsets.to(device), target.to(device)

            y_pred = model(text, offsets)

            loss = loss_func(y_pred, target)

            val_loss += loss.item()
            val_acc += (y_pred.argmax(1) == target).sum().item()
            count += batch_size

        print(f"{datetime.now()}: Epoch {epoch}: Val Acc {val_acc / count}: Val Loss {val_loss / count}")


@click.command()
@click.option("--dataset-file", required=True)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--num-epochs", default=100, type=int)
@click.option("--layer-size", default=32, type=int)
def main(dataset_file: str, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")

    dataset = DisasterTweetDataset.from_csv(dataset_file)
    vocab_size = len(dataset.tweet_vectorizer.vocab)

    model = LinearTweetClassifier(vocab_size, kwargs["layer_size"], n_classes=2)
    model.to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=kwargs["learning_rate"])

    trainer(
        dataset,
        model,
        optimizer,
        loss_func,
        kwargs["num_epochs"],
        kwargs["batch_size"],
        device=device,
    )
