import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def create_dataloader(train_dir: str,
                      test_dir: str,
                      train_transforms: transforms.Compose,
                      test_transforms: transforms.Compose,
                      batch_size: int = 32,
                      num_workers: int = os.cpu_count
                      ):

  # Create transforms
  train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
  ])

  test_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
  ])

  # Load and transform images
  train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
  test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

  # Retrieve the list of target classes
  class_names = train_data.classes

  # Create dataloaders
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=os.cpu_count(),
                                shuffle=True)

  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               num_workers=os.cpu_count(),
                               shuffle=False)

  return train_dataloader, test_dataloader, class_names
