import torch
import pandas as pd
from torchvision import transforms
from math import exp
from .dataset import ImageDataset
from ..data_stats import DataStats


class Predictions:
    @staticmethod
    def predict(model, data_dir, true_csv_path, size=[224, 224], csv_header=None):
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        dataset = ImageDataset(
            true_csv_path, data_dir, transform, csv_header=csv_header
        )
        batch_size = 64
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        model.eval()
        with torch.no_grad():
            pred_labels = []
            pred_probs = []
            for batch_idx, (filenames, batch_images, batch_labels) in enumerate(loader):
                batch_images = batch_images.to("cuda")
                batch_labels = batch_labels.to("cuda")
                output, class_labels = Predictions.predict_batch(
                    model, batch_images, no_grad=True
                )
                class_labels = class_labels.cpu().numpy().tolist()
                output = output.cpu().numpy().tolist()
                for i in range(len(filenames)):
                    pred = [filenames[i]] + class_labels[i]
                    pred_labels.append(pred)
                    pred = [filenames[i]] + [_sigmoid(o) for o in output[i]]
                    pred_probs.append(pred)

        return pd.DataFrame(pred_labels, columns=DataStats.columns), pd.DataFrame(
            pred_probs, columns=DataStats.columns
        )

    @staticmethod
    def predict_batch(model, batch_images, no_grad: bool, threshold=0.5):
        if no_grad:
            with torch.no_grad():
                output = model(batch_images)
        else:
            output = model(batch_images)

        class_labels = torch.where(output > threshold, 1, 0)
        return output, class_labels

    @staticmethod
    def save_predictions(pred_df, folder, filename):
        pred_df.to_csv(folder + filename, index=False, header=False, sep=" ")


def _sigmoid(x):
    return 1 / (1 + exp(-x))
