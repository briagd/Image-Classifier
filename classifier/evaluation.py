from .data_stats import DataStats

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from statistics import mean


class Eval:
    @staticmethod
    def evaluate(y_true, y_pred, file_path, drop=True):
        """
        Evaluates the predictions against true labels using prcision, recall and
        f1 scores. Saves the results in a file.

        Args:
            y_true: dataframe of the true lables
            y_pred: dataframe of the predicted labels
            file_path: path of the file where the results are saved
            drop: bool to specify if the first column of y_pred needs to be dropped

        """
        y_true = DataStats.drop_fname_col(y_true)
        y_pred = DataStats.drop_fname_col(y_pred)

        # Calculate precision, recall, and f1 score
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        auc = roc_auc_score(y_true, y_pred, average=None)
        eer = calculate_eer(y_true, y_pred)

        # Prepare for printing
        dash = "-" * 63
        with open(file_path, "a") as f:
            # Loop over classes
            for i in range(len(DataStats.classes) + 1):
                # Print the header
                if i == 0:
                    f.write(dash + "\n")
                    f.write(
                        "{:<15}{:<12}{:<9}{:<12}{:<10}{:<11}\n".format(
                            "Class", "precision", "recall", "f1 score", "EER", "AUC"
                        )
                    )
                    f.write(dash + "\n")
                # Print precision, recall and f1 score for each of the labels
                else:
                    f.write(
                        "{:<17}{:<11.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}\n".format(
                            DataStats.classes[i - 1],
                            precision[i - 1],
                            recall[i - 1],
                            f1[i - 1],
                            eer[i - 1],
                            auc[i - 1],
                        )
                    )

            # Print average precision
            precision_micro = precision_score(
                y_true, y_pred, average="micro", zero_division=0
            )
            f.write("{:<20}{:<4.2f}".format("\nAverage precision:", precision_micro))
            # Print average recall
            recall_micro = recall_score(
                y_true, y_pred, average="micro", zero_division=0
            )
            f.write("{:<19}{:<4.2f}".format("\nAverage recall:", recall_micro))
            # Print average f1 score
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
            f.write("{:<19}{:<12.2f}".format("\nAverage f1 score:", f1_micro))
            auc_micro = roc_auc_score(y_true, y_pred, average="micro")
            f.write("{:<19}{:<12.2f}".format("\nAverage EER:", mean(eer)))
            f.write("{:<19}{:<12.2f}".format("\nAverage AUC:", auc_micro))

    @staticmethod
    def eval_metrics(pred_labels, true_labels):
        """"""
        metrics = {
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": f1_score,
        }
        results = {
            name: metric_fn(true_labels, pred_labels, average="micro", zero_division=0)
            for name, metric_fn in metrics.items()
        }
        return results


# Adapted from https://github.com/scikit-learn/scikit-learn/issues/15247
# to support multilabel
def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classifier output.
    """
    eers = []
    for col_true, col_score in zip(y_true.columns, y_score.columns):
        fpr, tpr, thresholds = roc_curve(
            y_true[col_true], y_score[col_score], pos_label=1
        )
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eers.append(eer)
    return eers
