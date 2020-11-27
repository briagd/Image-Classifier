from .pytorch_models.resnext import Resnext
from .pytorch_models.resnest import Resnest
from .pytorch_models.predictions import Predictions
from .ensemble import Ensemble
from .evaluation import Eval
from .data_stats import DataStats
from torch import manual_seed, load
import pandas as pd


def main():
    manual_seed(3)
    # Train a model
    root_path = "../Challenge_train/"
    # model = Resnest()
    # model.train_model(
    #     root_path + "train.csv",
    #     root_path + "train",
    #     root_path + "val.csv",
    #     root_path + "train",
    #     num_epochs=30,
    #     batch_size=64,
    #     save_graph=True,
    #     graph_fname="resnest-loss.png",
    #     save_model=True,
    #     model_fname="resnest_model3",
    # )

    # model = Resnext()
    # model.train_model(
    #     root_path + "train.csv",
    #     root_path + "train",
    #     root_path + "val.csv",
    #     root_path + "train",
    #     num_epochs=10,
    #     batch_size=64,
    #     save_graph=True,
    #     graph_fname="resnext-loss4.png",
    #     save_model=True,
    #     model_fname="resnext_model4",
    # )

    # # Data to predict
    labels_file = "../Challenge_train/val.csv"
    data_dir = "../Challenge_train/train"
    # labels_file = "../Challenge_test/test.anno.txt"
    # data_dir = "../Challenge_test/test"
    #

    resnest_model = load("resnest_model")
    # make predictions
    resnest_pred_df, resnest_probs_df = Predictions.predict(
        resnest_model, data_dir, labels_file, csv_header=None
    )

    resnest_model3 = load("resnest_model3")
    # make predictions
    resnest3_pred_df, resnest3_probs_df = Predictions.predict(
        resnest_model3, data_dir, labels_file, csv_header=None
    )

    resnext_model = load("resnext_model")
    resnext_pred_df, resnext_probs_df = Predictions.predict(
        resnext_model, data_dir, labels_file, size=[299, 299], csv_header=None
    )

    resnext_model2 = load("resnext_model2")
    resnext2_pred_df, resnext2_probs_df = Predictions.predict(
        resnext_model2, data_dir, labels_file, size=[299, 299], csv_header=None
    )

    resnext_model3 = load("resnext_model3")
    resnext3_pred_df, resnext3_probs_df = Predictions.predict(
        resnext_model3, data_dir, labels_file, size=[299, 299], csv_header=None
    )
    # Get true labels
    y_true = pd.read_csv(labels_file, delimiter=" ", header=None)
    #
    # Evaluate models
    Eval.evaluate(y_true, resnext_pred_df, "resnext_eval.txt")
    Eval.evaluate(y_true, resnext2_pred_df, "resnext_eval2.txt")
    Eval.evaluate(y_true, resnext3_pred_df, "resnext_eva3.txt")
    Eval.evaluate(y_true, resnest_pred_df, "resnest_eval.txt")
    Eval.evaluate(y_true, resnest3_pred_df, "resnest_eval3.txt")
    #
    # Ensemble
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    prob_dfs = [
        resnext_probs_df,
        resnext2_probs_df,
        resnext3_probs_df,
        resnest_probs_df,
        resnest3_probs_df,
    ]
    # Using weighted sum of the probabilities
    ensemble_preds = Ensemble.compute_agg(prob_dfs, 0.6, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_probs.txt", drop=False)
    DataStats.confusion_matrix(
        ensemble_preds, y_true, save_fname="confusion_matrix.png"
    )

    # Using a simple vote
    pred_dfs = [
        resnext_pred_df,
        resnext2_pred_df,
        resnext3_pred_df,
        resnest_pred_df,
        resnest3_pred_df,
    ]
    ensemble_preds = Ensemble.compute_agg(pred_dfs, 3)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_simple_vote.txt", drop=False)

    # Using a weighted vote
    weights = [1, 1, 1, 1, 1]
    ensemble_preds = Ensemble.compute_agg(pred_dfs, 2, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_weighted_vote.txt", drop=False)
