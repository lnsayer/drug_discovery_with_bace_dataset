import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path
from typing import Callable, Optional, Any
import pickle
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm
from timeit import default_timer as timer


#--------------------------------------------------------------------train_step--------------------------------------------------------------------

def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  """
  Performs the training of a model for one epoch for the training dataloader.

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns a tuple of three lists (training loss, accuracy and AUC) of the training dataloader for the epoch.
  """

  model.to(device)
  model.train()

  train_loss, train_acc, train_auc = 0, 0, 0

  # We time how long it takes for each section in the training process
  auc_time = 0
  out_time = 0
  loss_time = 0
  optimizer_time = 0
  section_time = 0
  dataloader_loop_time = 0
  inside_loop_time = 0


  loop_start_time = timer()

  # Loop over the batches
  for idx, batch in enumerate(dataloader):
    # print(f"entered {idx} loop of train step")
    inside_loop_start_time = timer()
    # Time how long it takes to obtain an idx and batch of the dataloader
    if idx > 1:
      dataloader_loop_end_time = timer()
      dataloader_loop_time += dataloader_loop_end_time-dataloader_loop_start_time

    # Can time how long any chosen section takes to run
    section_start_time = timer()
    to_device_start_time = timer()
    batch = batch.to(device)
    to_device_end_time = timer()

    # Optimizer zero grad
    optimizer.zero_grad()

    # Forward pass
    out_start_time = timer()
    out = model(batch)
    out_end_time = timer()
    out_time+=out_end_time-out_start_time

    # Calculate the loss
    loss = loss_fn(out, batch.Class)
    train_loss += loss.item()

    # Calculate the label predictions
    label_preds = torch.argmax(out, dim=1)
    # Calculate accuracy
    train_acc += (label_preds == batch.Class).sum()

    # Calculate AUC
    auc_start_time = timer()
    # Check both classes present in batch.Class, otherwise add the batch_auc from the previous iteration
    if len(torch.unique(batch.Class)) == 2:
        batch_auc = roc_auc_score(batch.Class.detach().cpu().numpy(), out[:,1].detach().cpu().numpy())
        train_auc += batch_auc
    else:
      train_auc += batch_auc

    auc_end_time = timer()
    auc_time += auc_end_time-auc_start_time


    # Loss backward
    loss_start_time = timer()
    loss.backward()
    loss_end_time = timer()
    loss_time += loss_end_time-loss_start_time

    # Optimizer step
    optimizer_start_time = timer()
    optimizer.step()
    optimizer_end_time = timer()
    optimizer_time = optimizer_end_time-optimizer_start_time
    section_end_time = timer()
    section_time+=section_end_time-section_start_time
    dataloader_loop_start_time = timer()
    inside_loop_end_time = timer()
    inside_loop_time += inside_loop_end_time-inside_loop_start_time


  loop_end_time = timer()
  # print(f"Section time is {section_time:.4f}")
  # print(f"Dataloader loop time is {dataloader_loop_time:.4f}")

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss/len(dataloader.dataset)
  train_acc = train_acc/len(dataloader.dataset)
  train_auc = train_auc/len(dataloader)
  # print(f"AUC calculation time: {auc_time:.4f}s, Forward pass: {out_time:.4f}s, Loss time: {loss_time:.4f}, Optimizer time: {optimizer_time:.4f}, To device time: {to_device_end_time-to_device_start_time:.4f}\n")


  #print(f"Train outside loop time is {loop_end_time-loop_start_time:.4f}, inside loop time is {inside_loop_time:.4f}")

  return train_loss, train_acc, train_auc

#--------------------------------------------------------------------test_step------------------------------------------------------------------------

def test_step(model:torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device):

  """
  Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns a tuple of three lists (test loss, accuracy and AUC) of the test dataloader for the epoch.
  """

  # Put model in eval mode
  model.eval()

  test_loss, test_acc, test_auc = 0, 0, 0

  # Turn on torch inference manager
  with torch.inference_mode():
    # Loop through data batches
    for idx, batch in enumerate(dataloader):
      # print(f"entered test step {idx} batch loop")
      batch = batch.to(device)

      # Forward pass
      out = model(batch)

      # Calculate the loss
      loss = loss_fn(out, batch.Class)
      test_loss += loss.item()

      # Calculate the label predictions
      label_preds = torch.argmax(out, dim=1)
      # Calculate accuracy
      test_acc += (label_preds == batch.Class).sum()/len(label_preds)

      # Calculate the AUC
      if len(torch.unique(batch.Class)) == 2:
        batch_auc = roc_auc_score(batch.Class.detach().cpu().numpy(), out[:,1].detach().cpu().numpy())
        test_auc += batch_auc
      else:
        test_auc += batch_auc


    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    test_auc = test_auc/len(dataloader)

    return test_loss, test_acc, test_auc

#--------------------------------------------------------------------moving_average--------------------------------------------------------------------

def moving_average(values:list , window_size:int):
    """
    Calculates the simple moving average of the last window_size elements in a list of values.
    If len(values) < window_size it returns None.

    Args:
      values (List[float]): A list of numerical values.
      window_size (int): The number of elements to consider for the moving average.

    Returns a float of the average
    """
    if len(values) < window_size:
        return None
    return sum(values[-window_size:]) / window_size

#--------------------------------------------------------------------train-----------------------------------------------------------------------------

def train(model: torch.nn.Module,
          device: torch.device,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          repeat_iteration: int,
          nb_repeats: int, 
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
          epochs: int =5,
          model_save_path: str = None,
          window_size: int=10,
          patience: int=10):
  """
  Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    model_save_path : a string used to save the model's parameters. If no string is provided it is not saved.
    window_size: The window size on which to calculate the moving average. See the function moving average



  Returns a dictionary of results.
  """

  # 1. Create empty results dictionary
  results = {"epoch": [],
             "train_loss": [],
             "train_acc": [],
             "train_auc": [],
             "test_loss": [],
             "test_acc": [],
             "test_auc": [],
             "saved_epochs": [],
             "test_loss_mov_avg": [],
             "test_auc_mov_avg": []}
  # 2. Loop through training and testing steps for a number of epochs
  best_moving_loss_avg = float('inf')
  best_moving_auc_avg = 0

  # Loop over the number of epochs
  for i in tqdm(range(epochs)):

    start_time = timer()
    train_step_start_time = timer()
    train_loss, train_acc, train_auc = train_step(model,
                                       train_dataloader,
                                       loss_fn,
                                       optimizer,
                                       device)
    train_step_end_time = timer()
    test_step_start_time = timer()
    test_loss, test_acc, test_auc = test_step(model,
                                    test_dataloader,
                                    loss_fn,
                                    optimizer,
                                    device)
    test_step_end_time = timer()
    # print(f"Train step time is {train_step_end_time-train_step_start_time:.4f}s, Test step time is {test_step_end_time-test_step_start_time:.4f}s\n")

    # 3. Print out what's happening
    print(f"Epoch: {i}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test auc: {test_auc:.4f}")
    print(f"Repeat Iteration: {repeat_iteration} of {nb_repeats}")
    # 4. Update results dictionary
    append_start_time = timer()
    results["epoch"].append(i)
    results["train_loss"].append(round(train_loss, 4))
    results["train_acc"].append(round(train_acc.item(), 4))
    results["train_auc"].append(round(train_auc, 4))
    results["test_loss"].append(round(test_loss, 4))
    results["test_acc"].append(round(test_acc.item(), 4))
    results["test_auc"].append(round(test_auc, 4))
    append_end_time = timer()
    # print(f"append time is{append_end_time-append_start_time:.4f}")

    # 5. If model_save_path provided, save the model to its path based on whether test loss and test AUC have improved.
    """
    Once the number of epochs is greater than the window_size a current moving average is created of the last 'window_size'
    loss and AUC values. As long as these current metrics are higher than the current moving average, the model is saved.
    If the current metrics are not better than the current moving average for 'patience' epochs the training stops early.
    """

    save_timer_start = timer()
    if model_save_path:
      current_moving_loss_avg = moving_average(results["test_loss"], window_size)
      if current_moving_loss_avg is not None:
        results["test_loss_mov_avg"].append(round(current_moving_loss_avg, 4))
      else:
        results["test_loss_mov_avg"].append(None)


      current_moving_auc_avg = moving_average(results["test_auc"], window_size)
      if current_moving_auc_avg is not None:
        results["test_auc_mov_avg"].append(round(current_moving_auc_avg, 4))
      else:
        results["test_auc_mov_avg"].append(None)

      if current_moving_loss_avg is not None and current_moving_auc_avg is not None:
        if current_moving_loss_avg < best_moving_loss_avg and current_moving_auc_avg > best_moving_auc_avg:
          without_improvement_count = 0
          results["saved_epochs"].append(i)
          torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
                   f=model_save_path)
          print(f"Saved model at epoch {i} with current average test loss: {current_moving_loss_avg:.4f} and previous best: {best_moving_loss_avg:.4f}")
          print(f"Saved model at epoch {i} with current average AUC loss: {current_moving_auc_avg:.4f} and previous best: {best_moving_auc_avg:.4f}")
          best_moving_loss_avg = current_moving_loss_avg
          best_moving_auc_avg = current_moving_auc_avg

        else:
          without_improvement_count += 1
          print(f"Without_improvement_count: {without_improvement_count}")
        if without_improvement_count > patience:
          print("Early Stopping")
          break
    save_timer_end = timer()
    end_time  = timer()
    print(f"Epoch took {end_time-start_time:.2f} seconds")
    # print(f"Time to save loop : {save_timer_end-save_timer_start:.4f}")

  # 6. Return the filled results at the end of the epochs

  return results

#---------------------------------------------------------------------run_model_repeats---------------------------------------------------------------------


def run_model_repeats(model_callable: Callable[[], torch.nn.Module],
                      device: torch.device,
                      train_dataloader: torch.utils.data.DataLoader,
                      test_dataloader: torch.utils.data.DataLoader,
                      optimizer_: Callable[[], torch.optim.Optimizer],
                      criterion: torch.nn.Module,
                      models_directory: Path=None,
                      num_hidden_channels: int = 128,
                      pool_method: Any = global_mean_pool,
                      nb_epochs: int = 300,
                      nb_repeats: int = 1,
                      window_size: int = 10,
                      patience: int = 50,
                      **model_kwargs):
  """
    Runs multiple training sessions for the specified model and optionally saves the models and results.

    Args:
        model_callable: A callable that returns a PyTorch model instance to be trained and tested.
        device: A target device (e.g., "cuda" or "cpu") on which the model will be computed.
        train_dataloader: A DataLoader instance for the training dataset.
        test_dataloader: A DataLoader instance for the testing dataset.
        optimizer_: A callable that returns a PyTorch optimizer for minimizing the loss function.
        criterion: A PyTorch loss function to calculate the loss on both datasets.
        models_directory: A Path object representing the directory where the models and results will be saved.
                          If not provided, models and results are not saved.
        num_hidden_channels: An integer specifying the number of hidden channels in the model (used in naming saved files).
        pool_method: The pooling method used in the model (used in naming saved files).
        nb_epochs: An integer indicating the number of epochs to train the model in each run.
        nb_repeats: An integer specifying how many times to repeat the training and testing process.
        window_size: The window size for calculating the moving average, used for tracking improvements in validation performance.
        patience: An integer defining how many epochs to wait for improvement before early stopping.

    Returns:
        None
  """

  for i in range(nb_repeats):
    if models_directory:
      model_save_name = f"{i}_{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}.pth"
      model_save_path  = models_directory / model_save_name
    else:
      model_save_path = None
    model = model_callable(num_hidden_channels = num_hidden_channels,
                           pool_method = pool_method,
                           **model_kwargs)
    optimizer = optimizer_(model.parameters())

    results = train(model,
        device,
        train_dataloader,
        test_dataloader,
        optimizer,
        i+1,
        nb_repeats,
        criterion,
        epochs = nb_epochs,
        model_save_path = model_save_path,
        window_size=window_size,
        patience=patience)
    if models_directory:
      with open(models_directory/f"{i}_{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_results.pkl", 'wb') as f:
        print("Saved results of this model")
        pickle.dump(results, f)
    else:
      print("Did not save results of this model")