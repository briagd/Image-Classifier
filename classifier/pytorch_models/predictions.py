import torch
import pandas as pd
from torchvision import transforms
from .dataset import ImageDataset


class Predictions:
    @staticmethod
    def predict(model, data_dir, true_csv_path):
        transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        dataset = ImageDataset(true_csv_path, data_dir, transform)
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
                class_probs, class_labels = Predictions.predict_batch(
                    model, batch_images, no_grad=True
                )
                class_labels = class_labels.cpu().numpy().tolist()
                class_probs = class_probs.cpu().numpy().tolist()
                for i in range(len(filenames)):
                    pred = [filenames[i]] + class_labels[i]
                    pred_labels.append(pred)
                    pred = [filenames[i]] + class_probs[i]
                    pred_probs.append(pred)

        return pd.DataFrame(pred_labels), pd.DataFrame(pred_probs)

    @staticmethod
    def predict_batch(model, batch_images, no_grad: bool):
        if no_grad:
            with torch.no_grad():
                class_probs = model(batch_images)
        else:
            class_probs = model(batch_images)
        threshold = 0.5
        class_labels = torch.where(class_probs > threshold, 1, 0)
        return class_probs, class_labels

    @staticmethod
    def save_predictions(pred_df, folder, filename):
        pred_df.to_csv(folder + filename, index=False, header=False, sep=" ")
