from timeit import default_timer as timer
import_start_time = timer()
import subprocess
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
graphdata_start_time = timer()
from deepchem.feat.graph_data import GraphData
import deepchem as dc
graphdata_end_time = timer()
graphdata_time = graphdata_end_time-graphdata_start_time
import os
import pandas as pd
import os.path as osp
import pickle
from pathlib import Path
import_end_time = timer()

data_setup_module_imports_time = import_end_time- import_start_time

#---------------------------------------------------------------------MoleculeDataset---------------------------------------------------------------------

class MoleculeDataset(Dataset):
  def __init__(self, root, csv_file, transform=None, pre_transform=None, pre_filter=None):
    """
    Custom torch geometric Dataset class to store the samples and their corresponding labels

    Args:
    root : Path where the dataset should be stored. This folder is split
    into raw_dir (downloaded dataset) and processed_dir(processed data).
    csv_file : Desired name of the CSV file to be saved.
    transform, pre_transform, pre_filter : optional transforms
    """
    self.csv_file = csv_file
    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    """
    If this file exists in raw_dir, the download is not triggered/
    (the download function is not implemented here)
    """
    return self.csv_file

  @property
  def processed_file_names(self):
    """
    If these files are found in raw_dir, processing is skipped
    """
    self.data = pd.read_csv(self.raw_paths[0]).reset_index()

    return [f'data_{i}.pt' for i in list(self.data.index)]

  def download(self):
    """
    No need to download the csv file as it is already downloaded
    """
    pass

  def process(self):
    """
    Converts molecules with SMILES formats into PyTorch graphs. Uses Deepchem's MolGraphConvFeaturizer to create a graph
    and then convert that to a torch graph with to_pyg_graph. Saves these in the processed directory.
    """
    self.data = pd.read_csv(self.raw_paths[0]).reset_index()
    featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True)

    for idx, row in self.data.iterrows():
      # Featurize molecule and convert to torch graph
      smiles = row['mol']
      label = row['Class']
      pic50 = row['pIC50']

      out = featurizer.featurize(smiles)
      pyg_out = GraphData.to_pyg_graph(out[0])
      pyg_out.Class = torch.tensor([label])
      pyg_out.smiles = smiles
      pyg_out.pic50 = pic50

      # data = Data(x=pyg_out.x, edge_index=pyg_out.edge_index, edge_attr=pyg_out.edge_attr,
      #            y=torch.tensor([label]), dtype = torch.float)

      torch.save(pyg_out, osp.join(self.processed_dir, f'data_{idx}.pt'))

  def len(self):
    """
    Returns number of samples in the dataset
    """
    return len(self.processed_file_names)

  def get(self, idx):
    """
    Loads a single graph
    """
    data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    return data

NUM_WORKERS = os.cpu_count()

#-------------------------------------------------------------------create_dataloaders-------------------------------------------------------------------


def create_dataloaders(root_directory: str,
                       batch_size: int,
                       shuffled_indices_path: Path,
                       num_workers: int=NUM_WORKERS,
                       train_fraction: float=0.8):
  """
  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    root_directory: where the csv_file is stored
    batch_size: batch_size
    shuffled_indices_path: The path to the file containing pre-shuffled indices.
    num_workers: an integer for number of workers per dataloader
    train_fraction: a float for what proportion the training set should form of the whole dataset

  Returns:
    A tuple of (dataset, train_dataloader, test_dataloader)
  """

  dataset = MoleculeDataset(root = root_directory, csv_file = "bace.csv")

  with open(shuffled_indices_path, "rb") as f:   # Unpickling
    all_shuffled_indices = pickle.load(f)

  train_indices_proportion = int(train_fraction*len(dataset))

  train_indices = all_shuffled_indices[:train_indices_proportion]
  test_indices = all_shuffled_indices[train_indices_proportion:]

  train_dataset = dataset[train_indices]
  test_dataset = dataset[test_indices]


  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  return dataset, train_dataset, test_dataset, train_dataloader, test_dataloader