import HolterDataset as Dataset
from torch.utils.data import DataLoader, random_split
import model as rete
from torch import nn
import torch
import numpy as np
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"
device ="cpu"
cache_path = 'cache'
ecgs, ids = Dataset.load(cache_path)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch):
    train_loss, train_acc = 0, 0
    model.to(device)
    for signal,  _, class_label in data_loader:
        signal, class_label = signal.to(device), class_label.to(device)
        train_pred = model(signal)
        loss = loss_fn(train_pred, class_label)
        train_loss += loss

        p = torch.softmax(train_pred, dim=1)
        class_est = torch.argmax(p, dim=1)
        train_acc += accuracy_fn(y_true=class_label, y_pred=class_est)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    if epoch % 10 == 0:
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def run(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, NUM_LEADS, NUM_EPOCHS, NUM_SOGGETTI):
    dataset = Dataset.ECGDataset(ecgs[:NUM_SOGGETTI, :NUM_LEADS, :], ids[:NUM_SOGGETTI], fs=128, n_windows=NUM_FINETRE, seconds=NUM_SECONDI)
    train_ratio = 0.5
    test_ratio = 0.5
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_BATCH, shuffle=False)

    input_shape = NUM_LEADS
    output_shape = len(dataset.classes) 
    hidden_units = 32 

    model = rete.DeepECG_DUMMY(input_shape, hidden_units, output_shape)
    dummy_sig, _, _ = train_dataset[:]
    final = model(dummy_sig)

    model=rete.DeepECG(input_shape, hidden_units, output_shape, final).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    epochs = NUM_EPOCHS


    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f"-------\n Epoch: {epoch}")

        train_step(data_loader=train_loader, 
                  model=model, 
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy_fn=accuracy_fn,
                  device=device, 
                  epoch=epoch)

    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for signal, _, class_label in test_loader:
            signal, class_label = signal.to(device), class_label.to(device)
            test_pred = model(signal)
            test_loss += loss_fn(test_pred, class_label)

            p = torch.softmax(test_pred, dim=1)
            class_est = torch.argmax(p, dim=1)
            test_acc += accuracy_fn(y_true=class_label, y_pred=class_est)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_acc



accuracy = run(NUM_FINETRE=500, NUM_SECONDI=10, NUM_BATCH=16, NUM_LEADS=2, NUM_EPOCHS=200, NUM_SOGGETTI=70)
print(accuracy)






