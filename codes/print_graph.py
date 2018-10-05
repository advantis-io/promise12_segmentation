import matplotlib.pyplot as plt
import os
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


def plot_learning_performance(history, experiment_file):
    with_validation = True

    loss = list(history.history['loss'])
    epochs = list(range(1, len(loss)+1))

    _, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax2 = axes[0, 0]
    ax2.plot(epochs, history.history['loss'], label='Training')
    if with_validation:
        ax2.plot(epochs, history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    plt.xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    ax4 = axes[1, 0]
    ax4.set_ylim(ax2.get_ylim())
    ax4.plot(epochs, smooth_curve(history.history['loss']), label='Training')
    if with_validation:
        ax4.plot(epochs, smooth_curve(history.history['val_loss']), label='Validation')
    ax4.set_title('Loss EMA')
    plt.xlabel('Epochs')
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(experiment_file)

    plt.clf()
    plt.cla()
    plt.close()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points