import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import data_setup, model_builder, engine, utils
from pathlib import Path
import os


# Setup hyperparameters
NUM_EPOCHS = 20
NUM_WORKERS = os.cpu_count()
LEARNING_RATE = 0.001


# Instantiate directories
image_path = Path('./datasets')
train_dir = image_path / "train/training_set"
test_dir = image_path / "test/testing_set"
print(image_path)


# Instantiate transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Instantiate dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir=train_dir,
                                                                              test_dir=test_dir,
                                                                              train_transforms=train_transforms,
                                                                              test_transforms=test_transforms)


# Create a TinyVGG model
model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)


# Declare loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# Train model
model_result = engine.train(model=model,
                            train_data=train_dataloader,
                            test_data=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            epochs=NUM_EPOCHS)


# Plot loss and accuracy curves
utils.plot_curves(model_result)


# Save model
utils.save_model(model)
