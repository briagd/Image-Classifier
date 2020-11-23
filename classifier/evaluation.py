from sklearn.metrics import precision_score, recall_score, f1_score

classes = [
    "indoor",
    "outdoor",
    "person",
    "day",
    "night",
    "water",
    "road",
    "vegetation",
    "tree",
    "mountains",
    "beach",
    "buildings",
    "sky",
    "sunny",
    "partly_cloudy",
    "overcast",
    "animal",
]


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
        y_true = y_true.drop(0, axis=1)
        if drop:
            y_pred = y_pred.drop(0, axis=1)
        # Calculate precision, recall, and f1 score
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Prepare for printing
        dash = "-" * 45
        with open(file_path, "a") as f:
            # Loop over classes
            for i in range(len(classes) + 1):
                # Print the header
                if i == 0:
                    f.write(dash + "\n")
                    f.write(
                        "{:<15}{:<12}{:<9}{:<4}\n".format(
                            "Class", "precision", "recall", "f1 score"
                        )
                    )
                    f.write(dash + "\n")
                # Print precision, recall and f1 score for each of the labels
                else:
                    f.write(
                        "{:<17}{:<11.2f}{:<10.2f}{:<10.2f}\n".format(
                            classes[i - 1], precision[i - 1], recall[i - 1], f1[i - 1]
                        )
                    )

            # Print average precision
            precision_micro = precision_score(y_true, y_pred, average="micro")
            f.write("{:<20}{:<4.2f}".format("\nAverage precision:", precision_micro))
            # Print average recall
            recall_micro = recall_score(y_true, y_pred, average="micro")
            f.write("{:<19}{:<4.2f}".format("\nAverage recall:", recall_micro))
            # Print average f1 score
            f1_micro = f1_score(y_true, y_pred, average="micro")
            f.write("{:<19}{:<12.2f}".format("\nAverage f1 score:", f1_micro))

    @staticmethod
    def eval_metrics(pred_labels, true_labels, prefix):
        """"""
        metrics = {
            f"{prefix}precision": precision_score,
            f"{prefix}recall": recall_score,
            f"{prefix}f1_score": f1_score,
        }
        results = {
            name: metric_fn(true_labels, pred_labels, average="micro", zero_division=0)
            for name, metric_fn in metrics.items()
        }
        return results
