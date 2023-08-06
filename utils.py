from pathlib import Path
from typing import Tuple, Dict, List
import torchvision
import torch
import matplotlib.pyplot as plt


def save_model(model: torch.nn.Module):
  MODEL_PATH = Path('models')
  MODEL_PATH.mkdir(parents=True, exist_ok=True)

  MODEL_NAME = 'fruit_recognition_model.pth'
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

  print(f'Saving model to: {MODEL_SAVE_PATH}')
  torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def plot_curves(model_result: Dict[str, List[float]]):
  train_loss = model_result['train_loss']
  train_acc = model_result['train_acc']
  test_loss = model_result['test_loss']
  test_acc = model_result['test_acc']
  epochs = range(len(train_loss))

  plt.figure(figsize=(10, 15))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss Curve")
  plt.legend()
  plt.xlabel("Epochs")

  plt.figure(figsize=(10, 15))
  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_acc, label="train_acc")
  plt.plot(epochs, test_acc, label="test_acc")
  plt.title("Accuracy Curve")
  plt.legend()
  plt.xlabel("Epochs")


def eval_model(model: torch.nn.Module,
               image_path: str,
               class_names: List[str] = None,
               transform = None,
               ):
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  with Image.open(image_path) as f:
    target_image = transform(f)

  model.to(device)

  model.eval()
  with torch.inference_mode():
    target_image = target_image.unsqueeze(0)
    target_image_pred = model(target_image.to(device)).to(device)

  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
  target_image = target_image.squeeze()
  target_image = target_image.squeeze()

  plt.figure(figsize=(5, 8))
  plt.imshow(target_image.permute(2,1,0))
  if class_names:
    title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Probs: {target_image_pred_probs.max().cpu():.3f}"
  else:
    title = f"Pred: {target_image_pred_label.cpu()} | Probs: {target_image_pred_probs.max().cpu():.3f}"

  plt.title(title)
  plt.axis(False)

