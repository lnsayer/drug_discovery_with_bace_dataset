from timeit import default_timer as timer
import subprocess
import torch
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv, MLP, GINConv, global_max_pool, SAGPooling, TopKPooling, GINEConv
from torch.nn import Linear, ReLU, Dropout, Softmax
import torch.nn as nn
import torch.nn.functional as F

#----------------------------------------------------------------------GCNClassifier----------------------------------------------------------------------


class GCNClassifier(torch.nn.Module):
  """
  Standard GCN graph classifier. Uses the graph convolutional operator from PyTorch geometric.
  """
  def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, pool_method:torch_geometric.nn.pool):
    """

    Args:
      in_channels : number of features of the graph's nodes
      hidden_channels : the number of hidden neurons in the network. The "width" of the network
      out_channels : the number of output features, i.e 2 for classification.
      pool_method : the pooling method to obtain graph embedding from node embedding.
    """
    super().__init__()
    # Convolutional Layers
    self.conv1 = GCNConv(in_channels, hidden_channels)

    self.conv2 = GCNConv(hidden_channels, hidden_channels)

    self.conv3 = GCNConv(hidden_channels, hidden_channels)

    # Linear layer used in classification
    self.lin = Linear(hidden_channels, out_channels)

    # Pooling method
    self.pool_method = pool_method

  def forward(self, data):
    """
    Forward pass of the network
    Args:
      data : the input data containing node features, edge indices, and batch information
    Returns probabilities of the two classes (drug/not drug)
    """

    # Obtain node embeddings
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

    x = self.conv1(x, edge_index)
    x = F.leaky_relu(x)
    x = self.conv2(x, edge_index)
    x = F.leaky_relu(x)
    x = self.conv3(x, edge_index)

    # Aggregate node embeddings
    x = self.pool_method(x, batch)

    # Regularisation
    x = F.dropout(x)

    # Classification
    x = self.lin(x)

    # Softmax to get probabilities
    x = F.softmax(x, dim=1)

    return x

class GraphConvClassifier(GCNClassifier):
  """
  Same architecture as GCN Classifier however uses GraphConv layers
  """
  def __init__(self, in_channels:int, hidden_channels:int, out_channels:int,  pool_method:torch_geometric.nn.pool):
    super().__init__(in_channels, hidden_channels, out_channels, pool_method)
    self.conv1 = GraphConv(in_channels, hidden_channels)

    self.conv2 = GraphConv(hidden_channels, hidden_channels)

    self.conv3 = GraphConv(hidden_channels, hidden_channels)

    self.pool_method = pool_method

#--------------------------------------------------------------------GATCLassifier--------------------------------------------------------------------


class GATClassifier(torch.nn.Module):
  """
  GAT Convolutional graph classifier. Uses the graph attention operator from PyTorch geometric
  """
  def __init__(self, in_channels:int, hidden_channels:int, out_channels:int, heads:int, pool_method:torch_geometric.nn.pool,
               use_edge_attr:bool):
    """
    Args:
      in_channels : number of features of the graph's nodes.
      hidden_channels : the number of hidden neurons in the network. The "width" of the network.
      out_channels : the number of output features, i.e 2 for classification.
      heads : the number of multi-headed attentions.
      pool_method : the pooling method to obtain graph embedding from node embedding.
      use_edge_attr : boolean variable which determines whether to use the edge attributes of the graph.
    """
    super().__init__()
    # Convolutional Layers
    self.conv1 = GATConv(in_channels,
                         hidden_channels,
                         heads,
                         concat = True)
    self.conv2 = GATConv(hidden_channels*heads,
                         hidden_channels,
                         heads,
                         concat=True)
    self.conv3 = GATConv(hidden_channels*heads,
                         hidden_channels,
                         1,
                         concat=False)
    self.lin = Linear(hidden_channels, out_channels)

    # Pooling method
    self.pool_method = pool_method

    # Whether to use the edge attributes
    self.use_edge_attr = use_edge_attr

  def forward(self, data):
    """
    Forward pass of the network
    Args:
      data : the input data containing node features, edge indices, and batch information
    Returns probabilities of the two classes (drug/not drug) for the batch
    """

    # Obtain node embeddings
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

    # can use edge attributes
    if self.use_edge_attr:
      x = self.conv1(x, edge_index, edge_attr)
      x = F.leaky_relu(x)
      x = self.conv2(x, edge_index, edge_attr)
      x = F.leaky_relu(x)
      x = self.conv3(x, edge_index, edge_attr)

    # not using edge attributes
    else:
      x = self.conv1(x, edge_index)
      x = F.leaky_relu(x)
      x = self.conv2(x, edge_index)
      x = F.leaky_relu(x)
      x = self.conv3(x, edge_index)

    # Aggregate node embeddings
    x = self.pool_method(x, batch)

    # Regularisation
    x = F.dropout(x)

    # Classification
    x = self.lin(x)

    x = F.softmax(x, dim=1)

    return x

#-------------------------------------------------------------------GINConvClassifier-------------------------------------------------------------------

class GINConvClassifier(torch.nn.Module):
  """
  Applies the graph isomorphism operator
  """
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pool_method: torch_geometric.nn.pool):
    """
    Constructor method

    Args:
      in_channels : number of features of the graph's nodes
      hidden_channels : the number of hidden neurons in the network. The "width" of the network
      out_channels : the number of output features, i.e 2 for classification.
      num_layers : the number of layers of the multi-layer perceptron
      pool_method : the pooling method to obtain graph embedding from node embedding.
    """

    super().__init__()

    self.convs = torch.nn.ModuleList()
    self.conv = GINConv
    self.pool_method = pool_method

    # Create multiple GINConv layers as specified by num_layers
    for _ in range(num_layers):
      mlp = MLP([in_channels, hidden_channels, hidden_channels])
      self.convs.append(self.conv(nn=mlp, train_eps=False))
      in_channels = hidden_channels

    # Define the final MLP
    self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm = None, dropout = 0.5)

  def forward(self, data):
    """
    Forward pass of the network
    Args:
      data : the input data containing node features, edge indices, and batch information
    Returns probabilities of the two classes (drug/not drug) for the batch
    """
    x, edge_index, batch = data.x, data.edge_index, data.batch
    for conv in self.convs:
      x = conv(x, edge_index).relu()
    x = self.pool_method(x, batch)
    x = self.mlp(x)
    return F.softmax(x, dim=1)

#----------------------------------------------------------------------GINEConvClassifier----------------------------------------------------------------------

class GINEConvClassifier(torch.nn.Module):
  """
  Same as the GINConvClassifier, however also uses edge attributes of the graphs
  """
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pool_method: torch_geometric.nn.pool,
               use_edge_attr:bool, edge_dim:int):
    """
    Constructor method
    Args:
      in_channels : number of features of the graph's nodes
      hidden_channels : the number of hidden neurons in the network. The "width" of the network
      out_channels : the number of output features, i.e 2 for classification.
      num_layers : the number of layers of the multi-layer perceptron
      pool_method : the pooling method to obtain graph embedding from node embedding.
      use_edge_attr : boolean variable to determine whether will use the edge attributes or not.
      edge_dim : the dimensionality of the edge attributes for the graph's edges
    """
    super().__init__()

    self.convs = torch.nn.ModuleList()
    self.conv = GINEConv
    self.pool_method = pool_method
    self.use_edge_attr = use_edge_attr
    self.edge_dim = edge_dim

    for _ in range(num_layers):
      mlp = MLP([in_channels, hidden_channels, hidden_channels])
      self.convs.append(self.conv(nn=mlp, train_eps=False, edge_dim=self.edge_dim))
      in_channels = hidden_channels

    self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm = None, dropout = 0.5)

  def forward(self, data):
    """
    Forward pass of the network
      data : the input data containing node features, edge indices, and batch information
    Returns probabilities of the two classes (drug/not drug) for the batch
    """
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
    for conv in self.convs:
      if self.use_edge_attr:
        x = conv(x, edge_index, edge_attr).relu()
      else:
        x = conv(x, edge_index).relu()

    x = self.pool_method(x, batch)
    x = self.mlp(x)
    return F.softmax(x, dim=1)
  
#----------------------------------------------------------------------MODIFIED_GINEConvClassifier----------------------------------------------------------------------


class MODIFIED_GINEConvClassifier(torch.nn.Module):
  """
  Same as the GINConvClassifier, however also uses edge attributes of the graphs
  """
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers, pool_method: torch_geometric.nn.pool,
               use_edge_attr:bool, edge_dim:int):
    """
    Constructor method
    Args:
      in_channels : number of features of the graph's nodes
      hidden_channels : the number of hidden neurons in the network. The "width" of the network
      out_channels : the number of output features, i.e 2 for classification.
      num_layers : the number of layers of the multi-layer perceptron
      pool_method : the pooling method to obtain graph embedding from node embedding.
      use_edge_attr : boolean variable to determine whether will use the edge attributes or not.
      edge_dim : the dimensionality of the edge attributes for the graph's edges
    """
    super().__init__()

    self.convs = torch.nn.ModuleList()
    self.conv = GINEConv
    self.pool_method = pool_method
    self.use_edge_attr = use_edge_attr
    self.edge_dim = edge_dim

    for _ in range(num_layers):
      mlp = MLP([in_channels, hidden_channels, hidden_channels])
      self.convs.append(self.conv(nn=mlp, train_eps=False, edge_dim=self.edge_dim))
      in_channels = hidden_channels

    self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm = None, dropout = 0.5)

  def forward(self, data):
    """
    Forward pass of the network
      data : the input data containing node features, edge indices, and batch information
    Returns probabilities of the two classes (drug/not drug) for the batch
    """
    x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
    for conv in self.convs:
      if self.use_edge_attr:
        x = conv(x, edge_index, edge_attr).relu()
      else:
        x = conv(x, edge_index).relu()

    graph_embeddings = self.pool_method(x, batch)
    x = self.mlp(graph_embeddings)
    output = F.softmax(x, dim=1)

    return output, graph_embeddings