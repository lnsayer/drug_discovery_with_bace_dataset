# Time all the imports. data_setup takes a long time because it has to import GraphData each time. So does torch_geometric.

from timeit import default_timer as timer

total_import_time_start = timer()
start_time = timer()
import torch
from torch_geometric.nn import global_mean_pool
end_time = timer()
print(f"torch, torch_geometric.nn took {end_time-start_time:.4f}")

start_time = timer()
import data_setup
from data_setup import *
end_time = timer()
print(f"Imported data_setup.py, took {end_time - start_time:.4f}")

start_time = timer()
import models
from models import *
end_time = timer()
print(f"Imported models.py, took {end_time - start_time:.4f}")

start_time = timer()
import engine
from engine import *
end_time = timer()
print(f"Imported engine.py, took {end_time - start_time:.4f}")

start_time = timer()
import utils
from utils import *
end_time = timer()
print(f"Imported utils.py, took {end_time-start_time:.4f}")
print(os.getcwd())

total_import_time_end = timer()

start_time = timer()
dataset, train_dataset, test_dataset, train_dataloader, test_dataloader = create_dataloaders(root_directory = "graph_neural_networks/bace_dataset/data/",
                                                                batch_size = 32, shuffled_indices_path  = "/home/louis/Documents/personal_coding/graph_neural_networks/bace_dataset/data/dataset_indices_list")
end_time = timer()

print("-------------------------------------------------------------------------------------------------------------------------------------------------")

print(f"time to import modules in data_setup.py: {data_setup.data_setup_module_imports_time}")
print(f"time to import graphdata in data_setup.py: {data_setup.graphdata_time}")
print(f"Total import time: {total_import_time_end - total_import_time_start}")

print(f"Created dataset, train_dataloader, test_dataloader, took {end_time - start_time:.4f}s")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

bace_models_path = Path("/home/louis/Documents/personal_coding/graph_neural_networks/bace_dataset/models")
new_bace_ginconv_models_path = bace_models_path / "new_ginconv_models"


if new_bace_ginconv_models_path.is_dir():
  print(f"{new_bace_ginconv_models_path} is already a directory")
else:
  print(f"{new_bace_ginconv_models_path} is not a directory, creating one")
  new_bace_ginconv_models_path.mkdir(parents=True, exist_ok=True)


# Run repeats

run_model_repeats(model_callable = ginconv_callable,
                  device = device,
                  train_dataloader = train_dataloader,
                  test_dataloader = test_dataloader,
                  optimizer_ = adam_optimizer_callable,
                  criterion = torch.nn.CrossEntropyLoss(),
                  models_directory = new_bace_ginconv_models_path,
                  nb_epochs = 300,
                  nb_repeats = 5,
                  window_size = 10,
                  patience = 50,
                  num_features=30,
                  learning_rate = 0.0001,
                  num_hidden_channels=128,
                  num_out_channels=2,
                  pool_method=global_mean_pool, 
                  num_layers=3
                  )
