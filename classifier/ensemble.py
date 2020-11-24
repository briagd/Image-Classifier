class Ensemble:
    def compute_agg(pred_dfs, threshold, weights=None):
        """
        Use ensemble method to aggregate the predictions from different models.
        Can be used for simple voting, weighted vote.

        Args:
            pred_dfs: list of dataframes containing the predictions (can be either
            labels or probabilities)
            threshold: threshold value for a vote to be taken into account
                        in the case of voting it can be half of the number of models,
                        for probabilities it can be 0.5
            weights: list of weights of each model, the sum of the weights shou
        """
        for i in range(len(pred_dfs)):
            # drop the filename column
            pred_dfs[i] = pred_dfs[i].drop(0, axis=1)
            if weights:
                # multiply each value by weight
                pred_dfs[i] = pred_dfs[i] * weights[i]
        # sum all the values
        summed = pred_dfs[0]
        for i in range(1, len(pred_dfs)):
            summed = summed.add(pred_dfs[i])
        # replace values by label
        summed = summed.where(summed >= threshold, 0)
        summed = summed.where(summed < threshold, 1)

        return summed
