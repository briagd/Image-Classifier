import torch
from torchvision import models, transforms
from torch import nn
from .dataset import ImageDataset
from .train import Trainer
from ..plot_utils import PlotUtils


class Mobilenet:
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

        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        # Freeze layers
        for param in mobilenet_v2.parameters():
            param.requires_grad = False

        # Define thelast layers to be retrained
        mobilenet_v2.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 17),
        )
        mobilenet_v2 = mobilenet_v2.to("cuda")

        optimizer = torch.optim.Adam(
            mobilenet_v2.classifier.parameters(), lr=0.001, weight_decay=0.01
        )

        training_losses, validation_losses = Trainer.train(
            mobilenet_v2,
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
            self._save_model(mobilenet_v2, "", "mobilenet")

            torch.cuda.empty_cache()

    def _save_model(self, model, dir, filename):
        torch.save(model, dir + filename)
