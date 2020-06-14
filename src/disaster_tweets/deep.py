
import click
import torch
from tqdm import tqdm
from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from disaster_tweets.tweet_data import DisasterTweetDataset, SplitType


class LinearTweetClassifier(Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.fc1 = Linear(in_features=n_nodes, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = sigmoid(y_out)
        return y_out


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


def trainer(dataset: DisasterTweetDataset, model: Module, optimizer: Optimizer, loss_func, num_epochs: int, batch_size: int, shuffle: bool = True, drop_last: bool = True, device: str = "cpu"):
    train_state = {}
    for epoch in tqdm(range(num_epochs)):
        train_state["epoch_index"] = epoch

        dataset.set_split(SplitType.TRAIN)

        running_loss = RunningValue(batch_size)
        running_acc = RunningValue(batch_size)

        model.train()
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = {name: tensor.to(device) for name, tensor in batch_data.items()}

            optimizer.zero_grad()
            y_pred = model(x_in=batch_data['x_data'].float())

            loss = loss_func(y_pred, batch_data['y_target'].float())
            running_loss.add(loss.item())

            loss.backward()
            optimizer.step()

            running_acc.add(compute_accuracy(y_pred, batch_data['y_target']))

        train_state["train_loss"] = running_loss.value
        train_state["train_acc"] = running_acc.value

        dataset.set_split(SplitType.VAL)

        running_loss = RunningValue(batch_size)
        running_acc = RunningValue(batch_size)

        model.eval()
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = {name: tensor.to(device) for name, tensor in batch_data.items()}

            y_pred = model(x_in=batch_data['x_data'].float())

            loss = loss_func(y_pred, batch_data['y_target'].float())
            running_loss.add(loss.item())
            running_acc.add(compute_accuracy(y_pred, batch_data['y_target']))

        train_state["val_loss"] = running_loss.value
        train_state["val_acc"] = running_acc.value

        print(train_state)


@click.command()
@click.option("--dataset-file", required=True)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--num-epochs", default=100, type=int)
def main(dataset_file: str, **kwargs):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    dataset = DisasterTweetDataset.from_csv(dataset_file)
    n_features = len(dataset.tweet_vectorizer.vocab)

    model = LinearTweetClassifier(n_nodes=n_features)
    model.to(device)

    loss_func = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=kwargs["learning_rate"])

    trainer(dataset, model, optimizer, loss_func, kwargs["num_epochs"], kwargs["batch_size"], device=device)
