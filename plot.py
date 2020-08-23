import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs.csv")

step = df["step"].values
# df["entropy_loss"] = -1 * df["entropy_loss"]


def single_plot(ax, key, window):
    y_0 = df[key].values
    if window is None:
        y = y_0
    else:
        y = df[key].rolling(window).mean().shift(-window // 2).values
    out = ax.plot(step, y, label=key)

    if window is not None:
        c = out[0].get_color()
        y = y_0

        ax.plot(step, y, label=key, alpha=0.5, color=c)


def configure(ax, key, y_label, title, log):
    ax.set_xlabel("steps")

    if log:
        ax.set_yscale("log")

    if y_label is None and isinstance(key, str):
        ax.set_ylabel(key)

    elif y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    ax.minorticks_on()


def plot(keys, index=None, y_label=None, title=None, window=None, log=False):
    if isinstance(keys, list):
        if index is None:
            n_rows = len(keys)
            n_cols = 1
        else:
            n_rows, n_cols = index

        fig, ax = plt.subplots(n_rows, n_cols)

        for i, key in enumerate(keys):
            if n_rows * n_cols == 1:
                axes = ax
            else:
                axes = ax[i]
            single_plot(axes, keys[i], window)
            axes.legend()

            configure(axes, keys[i], y_label, title, log)

    if isinstance(keys, str):
        fig, ax = plt.subplots()
        single_plot(ax, keys, window)

        configure(ax, keys, y_label, title, log)


plot(["fake_loss", "real_loss"], index=(1, 1), log=True)
plot("D_loss", log=True)

plot("mean_discriminator_return", window=10, log=True)
plot("mean_environment_return", log=True)
plot("mean_episode_return")

plot(["total_loss", "pg_loss", "entropy_loss", "baseline_loss"])
plot("baseline_loss", log=True)
plot("n_discriminator_updates")

plt.show()
