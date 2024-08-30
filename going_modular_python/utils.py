
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from models import GCNClassifier, GINConvClassifier, GraphConvClassifier, GATClassifier, GINEConvClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle 
from typing import Any
import matplotlib.pyplot as plt



device = "cuda" if torch.cuda.is_available() else "cpu"

#---------------------------------------------------------------adam_optimizer_callable---------------------------------------------------------------


def adam_optimizer_callable(parameters, lr=0.001, weight_decay=0):
  """
  Creates an Adam optimizer with the specified parameters.

  Args:
    parameters (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize or dictionaries defining parameter groups.
    lr (float, optional): Learning rate. Default is 0.001.
    weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.

  Returns:
  torch.optim.Adam: Configured Adam optimizer instance.
  """
  return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

#---------------------------------------------------------------------gcn_callable---------------------------------------------------------------------

def gcn_callable(num_features, num_hidden_channels, num_out_channels, pool_method):
    """
    Returns an instance of the GCNClassifier model with specified architecture parameters.
    """
    return GCNClassifier(num_features, num_hidden_channels, num_out_channels, pool_method)

#---------------------------------------------------------------------ginconv_callable---------------------------------------------------------------------

def ginconv_callable(num_features, num_hidden_channels, num_out_channels, num_layers, pool_method):
    
    """
    Returns an instance of the GINConvClassifier model with specified architecture parameters.
    """

    return GINConvClassifier(num_features, num_hidden_channels, num_out_channels, num_layers, pool_method)

#---------------------------------------------------------------------gineconv_callable---------------------------------------------------------------------


def gineconv_callable(num_features, num_hidden_channels, num_out_channels, num_layers, pool_method, use_edge_attr, edge_dim):
    
    """
    Returns an instance of the GINConvClassifier model with specified architecture parameters.
    """

    return GINEConvClassifier(num_features, num_hidden_channels, num_out_channels, num_layers, pool_method, use_edge_attr, edge_dim)

#---------------------------------------------------------------------graphconv_callable---------------------------------------------------------------------

def graphconv_callable(num_features, num_hidden_channels, num_out_channels, pool_method):
    
    """
    Returns an instance of the GraphConvClassifier model with specified architecture parameters.
    """

    return GraphConvClassifier(num_features, num_hidden_channels, num_out_channels, pool_method)

#---------------------------------------------------------------------gat_callable---------------------------------------------------------------------

def gat_callable(num_features, num_hidden_channels, num_out_channels, heads, pool_method, use_edge_attr):
    """
    Returns an instance of the GATClassifier model with specified architecture parameters.
    """
    return GATClassifier(num_features, num_hidden_channels, num_out_channels, heads, pool_method, use_edge_attr)

    
#-----------------------------------------------------------------new_metric_func-----------------------------------------------------------------


def new_metric_func(model, train_dataloader, test_dataloader, batch_size = 32,threshold=0.5):
    """
    Calculates the metrics (AUC, Scikit-Learn's classification report metrics, 
    and confusion matrices) for a model on its training and test data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test data.
        threshold (float, optional): Threshold for converting predicted probabilities 
                                     to binary predictions. Default is 0.5.

    Returns:
        Tuple[float, float, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: 
            - Training AUROC
            - Test AUROC
            - Training classification report as a DataFrame
            - Test classification report as a DataFrame
            - Training confusion matrix (normalized) as a NumPy array
            - Test confusion matrix (normalized) as a NumPy array
    """
    with torch.inference_mode():
        model.eval()

        # Create empty tensors to fill with probabilities, predictions, and labels
        total_train_probs = torch.empty(len(train_dataloader.dataset))
        total_train_preds = torch.empty(len(train_dataloader.dataset))
        total_train_labels = torch.empty(len(train_dataloader.dataset))

        total_test_probs = torch.empty(len(test_dataloader.dataset))
        total_test_preds = torch.empty(len(test_dataloader.dataset))
        total_test_labels = torch.empty(len(test_dataloader.dataset))

        # Loop over batches and add to the total tensors
        for idx, batch in enumerate(train_dataloader):
            
            batch = batch.to(device)
            current_batch_size= len(batch)
            out = model(batch)
            train_probs = out[:, 1]
            train_preds = (train_probs >= threshold).long()

            total_train_probs[idx * batch_size:idx * batch_size + current_batch_size] = train_probs
            total_train_preds[idx * batch_size:idx * batch_size + current_batch_size] = train_preds
            total_train_labels[idx * batch_size:idx * batch_size + current_batch_size] = batch.Class

        for idx, batch in enumerate(test_dataloader):
            batch = batch.to(device)
            current_batch_size = len(batch)
            out = model(batch)
            test_probs = out[:, 1]
            test_preds = (test_probs >= threshold).long()

            total_test_probs[idx * batch_size:idx * batch_size + current_batch_size] = test_probs
            total_test_preds[idx * batch_size:idx * batch_size + current_batch_size] = test_preds
            total_test_labels[idx * batch_size:idx * batch_size + current_batch_size] = batch.Class

        # Calculate AUC and dataframes of metrics (using Scikit-Learn's metrics)

        
        train_auroc = roc_auc_score(total_train_labels, total_train_probs).item()
        train_classification_report = classification_report(total_train_labels, total_train_preds, output_dict=True)
        train_report_df = pd.DataFrame(data=train_classification_report).transpose()
        train_confusion_matrix = confusion_matrix(total_train_labels, total_train_preds, normalize='true')

        test_auroc = roc_auc_score(total_test_labels, total_test_probs).item()
        test_classification_report = classification_report(total_test_labels, total_test_preds, output_dict=True)
        test_report_df = pd.DataFrame(data=test_classification_report).transpose()
        test_confusion_matrix = confusion_matrix(total_test_labels, total_test_preds, normalize='true')

    return train_auroc, test_auroc, train_report_df, test_report_df, train_confusion_matrix, test_confusion_matrix
        
#---------------------------------------------------------------average_model_metrics---------------------------------------------------------------

def average_model_metrics(model: torch.nn.Module,
                          models_directory: Path,
                          model_name_stem: str,
                          repeats: int,
                          save_yes_no: bool, 
                          train_dataloader: torch.utils.data.DataLoader,
                          test_dataloader: torch.utils.data.DataLoader,
                          num_hidden_channels: int,
                          nb_epochs: int,
                          pool_method: Any, 
                          threshold: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Calculates and returns the average performance metrics for a repeated GNN model architecture.

    This function loads a specified number of trained model instances from a given directory, evaluates their performance
    on provided training and test datasets, and computes the mean and standard deviation of several performance metrics
    including AUROC and classification reports (e.g., precision, recall, F1-score) for both datasets.

    Args:
        model (torch.nn.Module): The neural network model instance used to load the parameters of the trained models.
        models_directory (Path): Directory where the trained model parameters are stored.
        model_name_stem (str): Common suffix for the model filenames (e.g., "_128_300_global_mean_pool.pth").
        repeats (int): Number of repeated model instances in the directory to load and evaluate.
        save_yes_no (bool): If True, saves the computed metrics (AUROC and classification reports) as pickle files in the specified directory.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        num_hidden_channels (int): Number of hidden channels in the model, used in the naming convention for saving results.
        nb_epochs (int): Number of epochs the model was trained for, used in the naming convention for saving results.
        pool_method (Any): Pooling method used in the model, used in the naming convention for saving results.
        threshold (float): Threshold value used for binary classification.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - DataFrame containing the mean and standard deviation of AUROC for both training and test datasets.
            - DataFrame containing the mean of the training performance metrics (e.g., precision, recall, F1-score).
            - DataFrame containing the mean of the test performance metrics (e.g., precision, recall, F1-score).
            - DataFrame containing the standard deviation of the training performance metrics.
            - DataFrame containing the standard deviation of the test performance metrics.

    The function performs the following steps:
    1. Loads the model parameters from the specified directory for each repeated instance.
    2. Evaluates the model on the provided training and test datasets using `new_metric_func`.
    3. Calculates the mean and standard deviation of the AUROC scores across all model instances.
    4. Computes the mean and standard deviation of the classification report metrics (e.g., precision, recall, F1-score).
    5. Optionally saves the computed metrics as pickle files for future analysis.
    """

    # Will append results of each model's performance to these lists (AUROC and classification report metric)
    train_auroc_list = []
    test_auroc_list = []
    train_report_list = []
    test_report_list = []

    # Loop over the number of repeats of the model runs
    for i in range(repeats):
        model_name = f"{i}" + model_name_stem
        model_path = models_directory / model_name
        model.load_state_dict(torch.load(f=model_path))
        model.to(device)
        train_auroc, test_auroc, train_classification_report, test_classification_report, train_confusion_matrix, test_confusion_matrix = new_metric_func(model, train_dataloader, test_dataloader, threshold=threshold)
        train_auroc_list.append(train_auroc)
        test_auroc_list.append(test_auroc)
        train_report_list.append(pd.DataFrame(train_classification_report))
        test_report_list.append(pd.DataFrame(test_classification_report))

    # Calculate averages and standard deviations of our repeats
    mean_train_auroc = np.mean(train_auroc_list)
    mean_test_auroc = np.mean(test_auroc_list)

    std_train_auroc = np.std(train_auroc_list)
    std_test_auroc = np.std(test_auroc_list)

    auroc_data = {"Train": [mean_train_auroc, std_train_auroc],
                "Test": [mean_test_auroc, std_test_auroc]}
    auroc_df = pd.DataFrame(auroc_data)


    mean_train_model_metrics = pd.DataFrame(pd.concat(train_report_list).groupby(level=0).mean())
    mean_test_model_metrics = pd.DataFrame(pd.concat(test_report_list).groupby(level=0).mean())

    std_train_model_metrics = pd.DataFrame(pd.concat(train_report_list).groupby(level=0).std())
    std_test_model_metrics = pd.DataFrame(pd.concat(test_report_list).groupby(level=0).std())

    # Option to save the results as pickle files (which can later be loaded and turned into dataframes)
    if save_yes_no:

        with open(models_directory/f"{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_auroc_df.pkl", 'wb') as f:
            print("Saved auroc dataframe")
            pickle.dump(auroc_df, f)

        with open(models_directory/f"{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_mean_train_metrics.pkl", 'wb') as f:
            print("Saved mean train metrics")
            pickle.dump(mean_train_model_metrics, f)

        with open(models_directory/f"{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_mean_test_metrics.pkl", 'wb') as f:
            print("Saved mean test metrics")
            pickle.dump(mean_test_model_metrics, f)

        with open(models_directory/f"{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_std_train_metrics.pkl", 'wb') as f:
            print("Saved std train metrics")
            pickle.dump(std_train_model_metrics, f)

        with open(models_directory/f"{num_hidden_channels}_{nb_epochs}_{pool_method.__name__}_std_test_metrics.pkl", 'wb') as f:
            print("Saved std test metrics")
            pickle.dump(std_test_model_metrics, f)


    return auroc_df, mean_train_model_metrics, mean_test_model_metrics, std_train_model_metrics, std_test_model_metrics

#-----------------------------------------------------------------loss_acc_auc_plots-----------------------------------------------------------------


def loss_acc_auc_plots(results:dict):
  """
  Plots the loss, accuracy, and AUROC metric curves for the training and test sets from the model results.

  Args:
    results (dict): Dictionary containing the model results. Expected keys are:
                    - "epoch": List of epoch numbers.
                    - "train_loss": List of training loss values.
                    - "test_loss": List of test loss values.
                    - "train_acc": List of training accuracy values.
                    - "test_acc": List of test accuracy values.
                    - "train_auc": List of training AUROC values.
                    - "test_auc": List of test AUROC values.
                    - "saved_epochs": List of epochs where the model was saved.

  Returns:
  None

  This function generates and displays three plots:
  1. Training and test loss over epochs.
  2. Training and test accuracy over epochs.
  3. Training and test AUROC over epochs.
  """

  fig, ax = plt.subplots(ncols=3, nrows=1, figsize = (15,6))

  ax[0].plot(results["epoch"], results["train_loss"], label="Train");
  ax[0].plot(results["epoch"], results["test_loss"],  label="Test");
  ax[0].vlines(results["saved_epochs"], ymin=0, ymax = max(results["test_loss"]), alpha=0.2, colors='black', linestyle='dashed', label = 'Saved epochs')

  ax[1].plot(results["epoch"], results["train_acc"], label="Train");
  ax[1].plot(results["epoch"], results["test_acc"],  label="Test");
  ax[1].vlines(results["saved_epochs"], ymin=0, ymax = max(results["train_acc"]), alpha=0.2, colors='black', linestyle='dashed', label = 'Saved epochs')

  ax[2].plot(results["epoch"], results["train_auc"], label="Train");
  ax[2].plot(results["epoch"], results["test_auc"],  label="Test");
  ax[2].vlines(results["saved_epochs"], ymin=0, ymax = max(results["train_auc"]), alpha=0.2, colors='black', linestyle='dashed', label = 'Saved epochs')

  ax[0].set_xlabel("Epochs", size=14)
  ax[0].set_ylabel("Loss", size=14)

  ax[1].set_xlabel("Epochs", size=14)
  ax[1].set_ylabel("Accuracy", size=14)

  ax[2].set_xlabel("Epochs", size=14)
  ax[2].set_ylabel("AUC", size=14)

  ax[0].legend();
  ax[1].legend(loc="lower right");
  ax[2].legend(loc = "lower right");

#--------------------------------------------------------------------plot_average_metrics--------------------------------------------------------------------

def plot_average_metrics(suffixes, models_paths_list, model_names_list, desired_metrics, bar_width, show_legend, fontsize, tick_labelsize):
    """
    Calculates and plots the average performance metrics for a list of GNN model architectures.

    This function loads performance metrics for multiple trained model instances from their respective directories,
    computes the mean and standard deviation of key metrics (such as AUROC, precision, recall, accuracy, f1-score, and support),
    and plots the results. The function returns DataFrames containing the calculated metrics and their associated errors.

    Args:
        suffixes (Dict[str, str]): A dictionary mapping metric types (e.g., 'auroc', 'mean_train') to their corresponding file suffixes.
                                   The function will search for files ending with these suffixes in the provided model paths.
        models_paths_list (List[Path]): A list of paths to the directories containing the trained model instances.
        model_names_list (List[str]): A list of strings to the name the bars on the bar plot
        desired_metrics (List[str]): A list of metrics to include in the plot, selected from the available metrics.
        bar_width (float): The width of the bars in the plot.
        show_legend (bool): Whether to display the legend in the plot.
        fontsize (int): The font size for the plot labels.
        tick_labelsize (int): The font size for the tick labels on the plot axes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame containing the average performance metrics (AUROC, precision, recall, accuracy, f1-score, support)
              for the test dataset across models.
            - DataFrame containing the standard errors for the same metrics.

    Example Usage:
        suffixes = {
            'auroc': "_auroc_df.pkl",
            'mean_train': "_mean_train_metrics.pkl",
            'mean_test': "_mean_test_metrics.pkl",
            'std_train': "_std_train_metrics.pkl",
            'std_test': "_std_test_metrics.pkl"
        }

        models_metrics_df, models_metrics_error_df = plot_average_metrics(
            suffixes,
            [bace_gcn_models_path, bace_gat_models_path, bace_graphconv_models_path, bace_ginconv_models_path, 
            bace_gat_edge_models_path, bace_gineconv_models_path],
            ["GCN", "GAT", "GraphConv", "GAT Edge", "GIN Conv", "GINE Conv"],
            ["precision", "recall", "accuracy"],
            bar_width=0.35,
            show_legend=True,
            fontsize=12,
            tick_labelsize=10
        )
    """
    # Initialize empty DataFrames to store metrics and errors
    metrics_df = pd.DataFrame()
    metrics_error_df = pd.DataFrame()

    # Define the metrics to extract
    metric_names = ["auroc", "precision", "recall", "accuracy", "f1-score", "support"]

    
    # Name the bars
    models_names_dictionaries = {}
    for i in range(len(models_paths_list)):
        models_names_dictionaries[models_paths_list[i]] = model_names_list[i]

    # Loop through each model's path and compute metrics
    for i, model_path in enumerate(models_paths_list):
        model_path = Path(model_path)

        # Function to find the correct file in the directory
        def find_file(suffix):
            files = list(model_path.glob(f"*{suffix}"))
            if len(files) != 1:
                raise ValueError(f"Expected one file ending with {suffix} in {model_path}, but found {len(files)}.")
            return files[0]

        # Find the correct files based on the suffixes
        auroc_file = find_file(suffixes['auroc'])
        mean_train_file = find_file(suffixes['mean_train'])
        mean_test_file = find_file(suffixes['mean_test'])
        std_train_file = find_file(suffixes['std_train'])
        std_test_file = find_file(suffixes['std_test'])

        # Load data
        auroc_df = pd.read_pickle(auroc_file)
        mean_train_metrics = pd.read_pickle(mean_train_file)
        mean_test_metrics = pd.read_pickle(mean_test_file)
        std_train_metrics = pd.read_pickle(std_train_file)
        std_test_metrics = pd.read_pickle(std_test_file)

        # Extract metrics and errors
        metrics = {
            "test_auroc": auroc_df["Test"].iloc[0],
            "test_precision": mean_test_metrics["precision"].iloc[-1],
            "test_recall": mean_test_metrics["recall"].iloc[-1],
            "test_accuracy": mean_test_metrics["precision"].loc["accuracy"],
            "test_f1_score": mean_test_metrics["f1-score"].iloc[-1],
            "test_support": mean_test_metrics["support"].iloc[-1]
        }

        errors = {
            "test_auroc": auroc_df["Test"].iloc[1] / (5**0.5),
            "test_precision": std_test_metrics["precision"].iloc[-1] / (5**0.5),
            "test_recall": std_test_metrics["recall"].iloc[-1] / (5**0.5),
            "test_accuracy": std_test_metrics["precision"].loc["accuracy"],
            "test_f1_score": std_test_metrics["f1-score"].iloc[-1] / (5**0.5),
            "test_support": std_test_metrics["support"].iloc[-1] / (5**0.5)
        }

        # Store metrics and errors
        model_name = models_names_dictionaries[models_paths_list[i]]
        metrics_df[model_name] = list(metrics.values())
        metrics_error_df[model_name] = list(errors.values())

    # Set index to the metric names
    metrics_df.index = metric_names
    metrics_error_df.index = metric_names

    # Plot metrics
    ax = metrics_df.loc[desired_metrics].plot.bar(
        yerr=metrics_error_df.loc[desired_metrics],
        figsize=(8, 4),
        width=bar_width,
        capsize=2
    )

    ax.tick_params(axis='both', labelsize=tick_labelsize)
    ax.set_ylabel("Metric Value", fontsize=fontsize)
    ax.set_xlabel("Models", fontsize=fontsize)

    if show_legend:
        ax.legend()
    plt.show()

    return metrics_df, metrics_error_df