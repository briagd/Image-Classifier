import torch
from torchvision import models, transforms
from torch import nn, optim
from .dataset import ImageDataset
from .train import Trainer
from ..plot_utils import PlotUtils


class Resnet:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.transform_train = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=10,
                    translate=None,
                    scale=None,
                    shear=10,
                    resample=0,
                    fillcolor=0,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.001, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self, dir: str, filename: str):
        return torch.load(dir + filename)

    def train_model(
        self,
        train_csv,
        train_data_dir,
        val_csv,
        val_data_dir,
        num_epochs=2,
        batch_size=64,
        save_graph=True,
        save_model=True,
    ):
        train_dataset = ImageDataset(train_csv, train_data_dir, self.transform_train)
        validation_dataset = ImageDataset(val_csv, val_data_dir, self.transform)
        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        resnet18 = models.resnet18(pretrained=True)
        # Freeze layers
        for param in resnet18.parameters():
            param.requires_grad = False
        # Define thelast layers to be retrained
        num_ftrs = resnet18.fc.in_features
        # Redifine last layer of the model
        resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 17),
        )
        resnet18 = resnet18.to("cuda")
        optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.05, momentum=0.9)
        training_losses, validation_losses = Trainer.train(
            resnet18,
            optimizer,
            train_loader,
            validation_loader,
            num_epochs=num_epochs,
            verbose=True,
        )

        if save_graph:
            PlotUtils.plot_losses(
                training_losses, validation_losses, "train-val-loss.png"
            )

        if save_model:
            self._save_model(resnet18, "", "resnetModel")
        torch.cuda.empty_cache()

    def _save_model(self, model, dir, filename):
        torch.save(model, dir + filename)
