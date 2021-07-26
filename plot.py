import matplotlib.pyplot as plt
import pandas as pd
import click


@click.group()
def cli():
    pass


def save(name):
    plt.savefig(f"figures/{name}.png")
    plt.savefig(f"figures/{name}.pdf")


@click.command()
def tree_performance():

    df = pd.read_csv("data/tree-performance.csv")
    print(df)

    fig, ax1 = plt.subplots(1, 1)
    for implementation in set(df.implementation):
        dfi = df[df.implementation == implementation]
        ax1.loglog(dfi.sample_size, dfi.time_mean, "-o", label=implementation)

    ax1.set_xlabel("Sample size")
    ax1.set_ylabel("CPU time")
    ax1.legend()
    save("tree-performance")


cli.add_command(tree_performance)

if __name__ == "__main__":
    cli()
