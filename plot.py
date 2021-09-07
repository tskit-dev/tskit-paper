import matplotlib.pyplot as plt
import pandas as pd
import click
import numpy as np


@click.group()
def cli():
    pass


def save(name):
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


def plot_tree_performance(name):
    df = pd.read_csv(f"data/{name}.csv")
    df = df[df.sample_size >= 1000]
    fig, ax = plt.subplots(1, 1)
    implementations = sorted(set(df.implementation))
    dfo = df[df.order == "msprime"]
    line_map = {}
    for implementation in implementations:
        dfi = dfo[dfo.implementation == implementation]
        (line,) = ax.loglog(dfi.sample_size, dfi.time_mean, "-o", label=implementation)
        line_map[implementation] = line

    dfo = df[df.order == "preorder"]
    for implementation in implementations:
        dfi = dfo[dfo.implementation == implementation]
        (line,) = ax.loglog(
            dfi.sample_size,
            dfi.time_mean,
            "--",
            color=line_map[implementation].get_color(),
        )

    legend1 = ax.legend()
    # TODO make both these lines black
    ax.legend(
        [line, list(line_map.values())[0]], ["preorder", "msprime"], loc="lower right"
    )
    ax.set_xlabel("Sample size")
    ax.set_ylabel("CPU Time")
    fig.add_artist(legend1)
    save(f"{name}")


@click.command()
def tree_performance():

    plot_tree_performance("tree-performance-sequential")
    plot_tree_performance("tree-performance-vectorised")


@click.command()
def tree_performance_relative():

    df1 = pd.read_csv("data/tree-performance-sequential.csv")
    df2 = pd.read_csv("data/tree-performance-vectorised.csv")
    df = pd.concat([df1, df2])

    print(df)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    dfo = df[df.order == "msprime"]
    implementations = sorted(set(df.implementation))
    print(dfo)
    norm = np.array(dfo[dfo.implementation == "c_lib"].time_mean)
    print(norm)
    line_map = {}
    ax = axes[0]
    ax.set_title("Node order = msprime")
    for implementation in implementations:
        if "py" not in implementation:
            dfi = dfo[dfo.implementation == implementation]
            (line,) = ax.plot(
                dfi.sample_size, dfi.time_mean / norm, "-o", label=implementation
            )
            line_map[implementation] = line

    ax = axes[1]
    ax.set_title("Node order = preorder")
    dfo = df[df.order == "preorder"]
    norm = np.array(dfo[dfo.implementation == "c_lib"].time_mean)
    for implementation in implementations:
        if "py" not in implementation:
            dfi = dfo[dfo.implementation == implementation]
            ax.plot(
                dfi.sample_size,
                dfi.time_mean / norm,
                "-o",
                color=line_map[implementation].get_color(),
            )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")

    legend1 = axes[0].legend()
    axes[0].set_ylabel("CPU Time relative to tskit C library call")
    plt.tight_layout()
    save("tree-performance-relative")


cli.add_command(tree_performance)
cli.add_command(tree_performance_relative)

if __name__ == "__main__":
    cli()
