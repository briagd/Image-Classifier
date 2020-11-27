import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns


class DataStats:

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

    columns = ["file_name"] + classes

    @staticmethod
    def class_counts(csv_path, csv_header=None):
        data_df = pd.read_csv(csv_path, sep=" ", header=csv_header)
        data_df = DataStats.drop_fname_col(data_df)

        return [data_df[col].sum() for col in data_df.columns]

    @staticmethod
    def pos_weights(csv_path, csv_header=None):
        class_counts = DataStats.class_counts(csv_path, csv_header)
        pos_weights = []
        neg_counts = [sum(class_counts) - pos_count for pos_count in class_counts]
        for i in range(len(class_counts)):
            pos_weights.append(neg_counts[i] / (class_counts[i] + 0.00001))
        return pos_weights

    @staticmethod
    def drop_fname_col(df):
        if 0 in df.columns:
            df = df.drop(0, axis=1)
        if "file_name" in df.columns:
            df = df.drop("file_name", axis=1)
        return df

    @staticmethod
    def confusion_matrix(
        pred_labels,
        true_labels,
        save_fname="confusion_matrix.png",
        title=None,
        drop=False,
    ):
        pred_labels = DataStats.drop_fname_col(pred_labels)
        true_labels = DataStats.drop_fname_col(true_labels)

        cm = multilabel_confusion_matrix(true_labels, pred_labels)
        # Normalise the values in each class
        cm = [class_cm / np.sum(class_cm) for class_cm in cm]

        fontsize = 8
        fig, axs = plt.subplots(5, 4, figsize=(12, 12))
        axs = axs.flatten()
        for ax in axs[len(DataStats.classes) :]:
            ax.remove()

        for axes, cfs_matrix, label in zip(axs, cm, DataStats.classes):
            df_cm = pd.DataFrame(
                cfs_matrix,
                index=["Y", "N"],
                columns=["Y", "N"],
            )

            heatmap = sns.heatmap(df_cm, cmap="plasma", annot=True, cbar=False, ax=axes)

            heatmap.yaxis.set_ticklabels(
                heatmap.yaxis.get_ticklabels(),
                rotation=0,
                ha="right",
                fontsize=fontsize,
            )
            heatmap.xaxis.set_ticklabels(
                heatmap.xaxis.get_ticklabels(),
                rotation=45,
                ha="right",
                fontsize=fontsize,
            )
            axes.set_xlabel("True label")
            axes.set_ylabel("Predicted label")
            axes.set_title(label)

        fig.tight_layout()
        plt.savefig(save_fname)
