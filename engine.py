import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  
  device = 'cuda' if torch.device.is_cuda_available() else 'cpu'

  ### Training
  train_loss, train_acc = 0, 0
  model.train()

  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)

    y_logits = model(X).to(device)

    loss = loss_fn(y_logits, y)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    # if batch % 400 == 0:
    #   print(f"Looked at {batch * len(X)} / {len(data_loader.dataset)} samples.")
    #   print(f"loss: {loss}")

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)

  print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")

  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
  
  device = 'cuda' if torch.device.is_cuda_available() else 'cpu'
  
  # Testing
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for (X, y) in data_loader:
      X, y = X.to(device), y.to(device)
      test_logits = model(X)
      test_loss += loss_fn(test_logits, y)
      test_pred_class = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
      test_acc += (test_pred_class == y).sum().item() / len(test_pred_class)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")
  print("---------------")

  return test_loss, test_acc


def train(model: torch.nn.Module,
          train_data: torch.utils.data.DataLoader,
          test_data: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 5):

  results = {
      "train_loss": 0,
      "train_acc": 0,
      "test_loss": 0,
      "test_acc": 0
  }

  for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n ----------")
    train_loss, train_acc = train_step(model=model,
                                       data_loader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer)

    test_loss, test_acc = test_step(model=model,
                                    data_loader=test_dataloader,
                                    loss_fn=loss_fn)

  results["train_loss"] = train_loss
  results["train_acc"] = train_acc
  results["test_loss"] = test_loss
  results["test_acc"] = test_acc

  return results
