from torch import nn
import numpy as np
from time import time
from .predictions import Predictions
from ..evaluation import Eval


class Trainer:
    @staticmethod
    def train(
        model,
        optimizer,
        training_loader,
        validation_loader,
        num_epochs,
        scheduler=None,
        verbose=True,
        pos_weights=None,
        device="cuda",
    ):
        # Loss, optimizer and metrics
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        training_losses = []
        validation_losses = []
        start_time = time()
        # Training
        for epoch in range(num_epochs):
            train_batch_losses = []
            train_true_labels = []
            train_pred_labels = []
            for batch_idx, (_, batch_images, batch_labels) in enumerate(
                training_loader
            ):
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                # make predictionson current batch
                output, class_labels = Predictions.predict_batch(
                    model, batch_images, no_grad=False
                )
                train_true_labels = (
                    train_true_labels + batch_labels.cpu().numpy().tolist()
                )
                train_pred_labels = (
                    train_pred_labels + class_labels.cpu().numpy().tolist()
                )
                # Loss computation
                loss = loss_function(output, batch_labels)
                train_batch_losses.append(loss.item())
                # SGD Step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation on validation set
            val_batch_losses = []
            val_true_labels = []
            val_pred_labels = []
            for batch_idx, (_, batch_images, batch_labels) in enumerate(
                validation_loader
            ):
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                val_output, val_class_labels = Predictions.predict_batch(
                    model, batch_images, no_grad=True
                )
                val_true_labels = val_true_labels + batch_labels.cpu().numpy().tolist()
                val_pred_labels = (
                    val_pred_labels + val_class_labels.cpu().numpy().tolist()
                )
                loss = loss_function(val_output, batch_labels)
                val_batch_losses.append(loss.item())

            # Add average loss over each batch to arrays
            mean_train_loss = np.mean(train_batch_losses)
            training_losses.append(mean_train_loss)
            mean_val_loss = np.mean(val_batch_losses)
            validation_losses.append(mean_val_loss)

            if scheduler:
                scheduler.step()

            # Print the different metrics and loss on train and val set
            if verbose:
                train_metrics = Eval.eval_metrics(
                    train_pred_labels,
                    train_true_labels,
                )
                train_metrics["loss"] = mean_train_loss

                val_metrics = Eval.eval_metrics(
                    val_pred_labels,
                    val_true_labels,
                )
                val_metrics["loss"] = mean_val_loss
                # print(f"epoch: {epoch}")
                # print(train_metrics, val_metrics)
                _output(epoch, train_metrics, val_metrics, time() - start_time)
        return training_losses, validation_losses


def _output(epoch, train_metrics, val_metrics, time):
    metrics = ["loss", "precision", "recall", "f1_score"]
    print("epoch: {:}, {:.1f} s".format(epoch, time))
    print("{:<15}{:<15}{:<15}".format("", "Training", "Validation"))
    for metric in metrics:
        print(
            "{:<15}{:<15.5f}{:<15.5f}".format(
                metric, train_metrics[metric], val_metrics[metric]
            )
        )
    return 0
