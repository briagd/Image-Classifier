import matplotlib.pyplot as plt


class PlotUtils:
    @staticmethod
    def plot_losses(training_losses, validation_losses, fname, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(training_losses, color="tab:blue", label="training")
        ax.plot(validation_losses, color="tab:orange", label="validation")
        if title:
            ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        plt.legend()
        plt.savefig(fname)
