import ruamel.yaml as yaml
import pandas as pd
import dataclasses



@dataclasses.dataclass
class Author:
    name: str
    affiliations: list[str]
    annotation: str = None

    def full_name(self):
        last, others = self.name.split(",", maxsplit=1)
        return f"{others} {last}"

@dataclasses.dataclass
class Annotation:
    symbol: str
    text: str


def main():
    with open("contributors.yaml") as f:
        d = yaml.YAML(typ="rt").load(f)

    df = pd.DataFrame(d["contributors"])
    print(df[df["affiliations"].isnull()])

    print(df["contribution"].value_counts())
    df = df[df["contribution"] != "trivial"]
    print("Total authors = ", df.shape[0])

    authors = []
    # first authors
    dfs = df[df["class"] == "first"]  # Keep the first author sorting
    annotation = Annotation("$\\ast$", "Joint first author")
    annotations = [annotation]
    for _, row in dfs.iterrows():
        authors.append(
            Author(row["name"], row["affiliations"], annotation=annotation)
        )
    print("len = ", len(authors))
    # Second authors
    dfs = df[df["contribution"] == "substantial"].sort_values("name")
    annotation = Annotation("$\\dagger$", "Joint second author")
    annotations.append(annotation)
    for _, row in dfs.iterrows():
        authors.append(
            Author(row["name"], row["affiliations"], annotation=annotation)
        )
    print("len = ", len(authors))
    # Middle authors
    dfs = df[df["contribution"] == "minor"].sort_values("name")
    for _, row in dfs.iterrows():
        authors.append(Author(row["name"], row["affiliations"]))
    print("len = ", len(authors))

    # last authors
    dfs = df[df["class"] == "last"]  # Keep the last author sorting
    annotation = Annotation("$\\ddagger$", "Joint senior author")
    annotations.append(annotation)
    for _, row in dfs.iterrows():
        authors.append(
            Author(row["name"], row["affiliations"], annotation=annotation)
        )

    assert len(authors) == df.shape[0]

    # Resolve the affiliations
    affil_map = {}
    for author in authors:
        for affil in author.affiliations:
            if affil not in affil_map:
                affil_map[affil] = len(affil_map) + 1

    for author in authors:
        affils = [str(affil_map[affil]) for affil in author.affiliations]
        if author.annotation is not None:
            affils.append(author.annotation.symbol)
        print("\\author[" + ",".join(affils) + "]{" + author.full_name() + "}")
    for affil, index in affil_map.items():
        print("\\affil[" + str(index) + "]{" + affil + "}")

    for annotation in annotations:
        print("\\affil[" + annotation.symbol + "]{" + annotation.text + "}")


if __name__ == "__main__":
    main()
