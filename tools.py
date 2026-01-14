
import ruamel.yaml as yaml
import pandas as pd

def main():
    with open("tools.yaml") as f:
       d = yaml.YAML(typ='rt').load(f)

    df = pd.DataFrame(d)
    print(df)

if __name__ == "__main__":
    main()
