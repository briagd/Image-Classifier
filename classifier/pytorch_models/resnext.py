from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim, save, load, cuda, log, as_tensor
from .dataset import ImageDataset
from .train import Trainer
from .transformations import Transformations

from ..plot_utils import PlotUtils
from ..data_stats import DataStats


class Resnext:
    def __init__(self, device="cuda"):

        self.device = device
        self.transform = Transformations(224)

    def load_model(self, dir: str, filename: str):
        return load(dir + filename)

    def train_model(
        self,
        train_csv,
        train_data_dir,
        val_csv,
        val_data_dir,
        num_epochs=2,
        batch_size=64,
        save_graph=True,
        graph_fname="train-val-loss.png",
        save_model=True,
        model_fname="resnext_model",
    ):
        train_dataset = ImageDataset(train_csv, train_data_dir, self.transform.training)
        validation_dataset = ImageDataset(
            val_csv, val_data_dir, self.transform.validation
        )
        batch_size = batch_size
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        resnext = models.resnext50_32x4d(pretrained=True)

        # Freeze layers
        for param in resnext.parameters():
            param.requires_grad = False
        # Define the last layer to be retrained
        resnext.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 17),
        )
        resnext = resnext.to(self.device)
        optimizer = optim.SGD(resnext.fc.parameters(), lr=0.1, momentum=0.9)

        # Weights for imbalanced data
        pos_weights = DataStats.pos_weights(train_csv)
        pos_weights = log(as_tensor(pos_weights, dtype=float))
        pos_weights = pos_weights.to(self.device)

        training_losses, validation_losses = Trainer.train(
            resnext,
            optimizer,
            train_loader,
            validation_loader,
            num_epochs=num_epochs,
            verbose=True,
            pos_weights=pos_weights,
            device=self.device,
        )

        if save_graph:
            PlotUtils.plot_losses(training_losses, validation_losses, graph_fname)

        if save_model:
            self._save_model(resnext, "", model_fname)

        if self.device == "cuda":
            cuda.empty_cache()

    def _save_model(self, model, dir, filename):
        save(model, dir + filename)
