from .pytorch_models.resnet import Resnet
from .pytorch_models.mobilenet import Mobilenet
from .pytorch_models.predictions import Predictions
from .ensemble import Ensemble
from .evaluation import Eval
import pandas as pd


def main():
    """
    # Train a model
    root_path = "../Challenge_train/"
    model = Mobilenet()
    model.train_model(
        root_path + "train.csv",
        root_path + "train",
        root_path + "val.csv",
        root_path + "train",
        num_epochs=1,
        batch_size=64,
        save_graph=True,
        save_model=False,
    )
    """

    # Data to predict
    labels_file = "../Challenge_train/val.csv"
    data_dir = "../Challenge_train/train"

    # Load Resnet model
    resnet = Resnet()
    resnet_model = resnet.load_model(dir="", filename="resnetModel")
    # make predictions
    resnet_pred_df, resnet_probs_df = Predictions.predict(
        resnet_model,
        data_dir,
        labels_file,
    )

    # Load Mobilenet model
    mobilenet = Mobilenet()
    # make predictions
    mobilenet_model = mobilenet.load_model(dir="", filename="mobilenet")
    mobilenet_pred_df, mobilenet_probs_df = Predictions.predict(
        mobilenet_model,
        data_dir,
        labels_file,
    )

    # Get true labels
    y_true = pd.read_csv(labels_file, delimiter=" ", header=None)
    # Evaluate models
    Eval.evaluate(y_true, resnet_pred_df, "resnet_eval.txt")
    Eval.evaluate(y_true, mobilenet_pred_df, "mobile_eval.txt")

    # Ensemble
    weights = [0.3, 0.7]
    prob_dfs = [resnet_probs_df, mobilenet_probs_df]
    # Using weighted sum of the probabilities
    ensemble_preds = Ensemble.compute_agg(prob_dfs, 0.5, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_probs.txt", drop=False)

    # Using a simple vote
    pred_dfs = [resnet_pred_df, mobilenet_pred_df]
    ensemble_preds = Ensemble.compute_agg(pred_dfs, 1)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_simple_vote.txt", drop=False)

    # Using a weighted vote
    pred_dfs = [resnet_pred_df, mobilenet_pred_df]
    ensemble_preds = Ensemble.compute_agg(pred_dfs, 1, weights)
    Eval.evaluate(y_true, ensemble_preds, "ensemble_weighted_vote.txt", drop=False)
