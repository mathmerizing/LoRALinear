# Simple image classification in PyTorch (https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
from tqdm import tqdm

from networks import NeuralNetwork, LoRANeuralNetwork, AENeuralNetwork

def get_dataloaders():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer):
    loss_history = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            loss_history.append(loss)
    return loss_history

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100.*correct, test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Low-rank approximation for neural networks')
    parser.add_argument("--nn", type=str, default="mlp", help="MLP or MLP+LoRa or MLP+AE", choices=["mlp", "mlp+lora", "mlp+ae"])
    parser.add_argument('--rank', type=int, default=8, help='Rank of the low-rank approximation')
    args = parser.parse_args()

    NN = args.nn
    rank = args.rank

    # print the arguments
    print("Setup:")
    print(f"  NN:     '{NN}'")
    print(f"  rank:    {rank}")

    # get the data loaders
    train_dataloader, test_dataloader = get_dataloaders()

    # get device for training
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: '{device}'")

    EPOCHS = 100
    print(f"  epochs:  {EPOCHS}")

    # define the model
    if NN == "mlp":
        model = NeuralNetwork().to(device)
    elif NN == "mlp+lora":
        model = LoRANeuralNetwork(rank=rank).to(device)
    elif NN == "mlp+ae":
        model = AENeuralNetwork(rank=rank).to(device)
    else:
        raise ValueError(f"Unknown neural network architecture '{NN}'")
    
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters\n\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train the neural network
    loss_history = []
    accuracy = []
    test_loss = []
    for t in tqdm(range(EPOCHS)):
        print(f"Epoch {t+1}\n-------------------------------")
        _losses = train(train_dataloader, model, loss_fn, optimizer)
        loss_history.extend(_losses)
        _acc, _test_loss = test(test_dataloader, model, loss_fn)
        accuracy.append(_acc)
        test_loss.append(_test_loss)
    
    # print the training history
    print("Training history:")
    print(f"{loss_history=}")
    print(f"{accuracy=}")
    print(f"{test_loss=}")