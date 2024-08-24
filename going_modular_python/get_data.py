from timeit import default_timer as timer
import requests
from pathlib import Path
import pandas as pd

project_path = Path("/home/louis/Documents/personal_coding/graph_neural_networks/bace_dataset/")
data_path = project_path / "data"
bace_path = data_path / "raw"


# Create directory and download bace.csv from my Github
if bace_path.is_dir():
  print(f"{bace_path} is already a directory")
else:
  print(f"{bace_path} is not a directory, creating one")
  bace_path.mkdir(parents=True, exist_ok=True)

  with open(bace_path / "bace.csv", "wb") as f:
    request = requests.get("https://raw.githubusercontent.com/lnsayer/personal_repo/main/drug%20discovery%20with%20BACE%20dataset/data/bace.csv")
    print("Downloading data")
    f.write(request.content)

# Resave the csv files without unnecessary columns
bace_df = pd.read_csv(bace_path/ "bace.csv")
bace_df = bace_df[["mol", "CID", "Class", "Model", "pIC50"]]
bace_df.to_csv(bace_path/"bace.csv")
print("Resaved bace_df without unnecessary columns")