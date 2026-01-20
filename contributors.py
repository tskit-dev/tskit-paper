
import ruamel.yaml as yaml
import pandas as pd

# def write_latex_output(authors):
#     for author in authors:

def main():
    with open("contributors.yaml") as f:
        d = yaml.YAML(typ="rt").load(f)

    df = pd.DataFrame(d["contributors"])
    print(df["contribution"].value_counts())
    df = df[df["contribution"] != "trivial"]
    print("Total authors = ", df.shape[0])

    # first authors
    dfs = df[df["class"] == "first"].sort_values("name")
    print(dfs)

    for contribution in ["substantial", "minor"]:
        dfs = df[df["contribution"] == contribution]
        print(f"===={contribution}==={dfs.shape[0]}")
        print(dfs["name"])
    # print("Total = ", df.shape[0])
    # df = df[~df["publication"].isnull()]
    # print(df[["name", "type", "interface", "package"]].sort_values("type"))

if __name__ == "__main__":
    main()
