from .pytorch_models.resnext import Resnext
from .pytorch_models.resnest import Resnest
from .pytorch_models.predictions import Predictions
from .ensemble import Ensemble
from .evaluation import Eval
from .data_stats import DataStats
from torch import manual_seed, load
import pandas as pd


def train():
    for i in range(3):
        manual_seed(i)
        # Train a model
        root_path = "../Challenge_train/"
        model = Resnest()
        model.train_model(
            root_path + "train.csv",
            root_path + "train",
            root_path + "val.csv",
            root_path + "train",
            num_epochs=30,
            batch_size=64,
            save_graph=True,
            graph_fname=f"resnest-loss-seed{i}.png",
            save_model=True,
            model_fname=f"resnest-model-seed{i}.pt",
        )

        model = Resnext()
        model.train_model(
            root_path + "train.csv",
            root_path + "train",
            root_path + "val.csv",
            root_path + "train",
            num_epochs=10,
            batch_size=64,
            save_graph=True,
            graph_fname=f"resnext-loss-seed{i}.png",
            save_model=True,
            model_fname=f"resnext-model-seed{i}.pt",
        )


def evaluate_validation():
    # Data to predict
    labels_file = "../Challenge_train/val.csv"
    data_dir = "../Challenge_train/train"
    header = None

    resnest_model = load("resnest_model")
    # make predictions
    resnest_pred_df, resnest_probs_df = Predictions.predict(
        resnest_model, data_dir, labels_file, csv_header=header
    )

    resnest_model3 = load("resnest_model3")
    # make predictions
    resnest3_pred_df, resnest3_probs_df = Predictions.predict(
        resnest_model3, data_dir, labels_file, csv_header=header
    )

    resnext_model = load("resnext_model")
    resnext_pred_df, resnext_probs_df = Predictions.predict(
        resnext_model, data_dir, labels_file, size=[299, 299], csv_header=header
    )

    resnext_model2 = load("resnext_model2")
    resnext2_pred_df, resnext2_probs_df = Predictions.predict(
        resnext_model2, data_dir, labels_file, size=[299, 299], csv_header=header
    )

    resnext_model3 = load("resnext_model3")
    resnext3_pred_df, resnext3_probs_df = Predictions.predict(
        resnext_model3, data_dir, labels_file, size=[299, 299], csv_header=header
    )
    y_true = pd.read_csv(labels_file, delimiter=" ", header=header)

    # Evaluate models
    Eval.evaluate(y_true, resnext_pred_df, "resnext_eval_val.txt")
    Eval.evaluate(y_true, resnext2_pred_df, "resnext_eval2_val.txt")
    Eval.evaluate(y_true, resnext3_pred_df, "resnext_eva3_val.txt")
    Eval.evaluate(y_true, resnest_pred_df, "resnest_eval_val.txt")
    Eval.evaluate(y_true, resnest3_pred_df, "resnest_eval3_val.txt")

    # Ensemble
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    probs = [
        resnext_probs_df,
        resnext2_probs_df,
        resnext3_probs_df,
        resnest_probs_df,
        resnest3_probs_df,
    ]
    # Using weighted sum of the probabilities
    ensemble_preds = Ensemble.compute_agg(probs, 0.6, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_probs_val.txt", drop=False)
    DataStats.confusion_matrix(
        ensemble_preds, y_true, save_fname="confusion_matrix_val.png"
    )


def evaluate_test():
    labels_file = "../Challenge_test/test.anno.txt"
    data_dir = "../Challenge_test/test"
    header = "infer"

    resnest_model = load("resnest_model")
    # make predictions
    resnest_pred_df, resnest_probs_df = Predictions.predict(
        resnest_model, data_dir, labels_file, csv_header=header
    )

    resnest_model3 = load("resnest_model3")
    # make predictions
    resnest3_pred_df, resnest3_probs_df = Predictions.predict(
        resnest_model3, data_dir, labels_file, csv_header=header
    )

    resnext_model = load("resnext_model")
    resnext_pred_df, resnext_probs_df = Predictions.predict(
        resnext_model, data_dir, labels_file, size=[299, 299], csv_header=header
    )

    resnext_model2 = load("resnext_model2")
    resnext2_pred_df, resnext2_probs_df = Predictions.predict(
        resnext_model2, data_dir, labels_file, size=[299, 299], csv_header=header
    )

    resnext_model3 = load("resnext_model3")
    resnext3_pred_df, resnext3_probs_df = Predictions.predict(
        resnext_model3, data_dir, labels_file, size=[299, 299], csv_header=header
    )

    # Get true labels
    y_true = pd.read_csv(labels_file, delimiter=" ", header=header)
    #
    # Evaluate models
    Eval.evaluate(y_true, resnext_pred_df, "resnext_eval_test.txt")
    Eval.evaluate(y_true, resnext2_pred_df, "resnext_eval2_test.txt")
    Eval.evaluate(y_true, resnext3_pred_df, "resnext_eva3_test.txt")
    Eval.evaluate(y_true, resnest_pred_df, "resnest_eval_test.txt")
    Eval.evaluate(y_true, resnest3_pred_df, "resnest_eval3_test.txt")

    Ensemble
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    probs = [
        resnext_probs_df,
        resnext2_probs_df,
        resnext3_probs_df,
        resnest_probs_df,
        resnest3_probs_df,
    ]

    # Using weighted sum of the probabilities
    ensemble_preds = Ensemble.compute_agg(probs, 0.6, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_probs_test.txt", drop=False)
    DataStats.confusion_matrix(
        ensemble_preds, y_true, save_fname="confusion_matrix_test.png"
    )


def main(args):
    if args[0] == "-train":
        train()
    elif args[0] == "-eval_test":
        evaluate_test()
    elif args[0] == "-eval_val":
        evaluate_validation()
    else:
        print("Argument -train, -eval_test or -eval_val required.")
