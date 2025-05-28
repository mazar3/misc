Okay, I understand the strategy! This is a good approach for exam preparation. I will go through each TP, extract the `def` functions and relevant Pytorch `Module` or `LightningModule` classes, and structure them as requested.

I'll aim for:
1.  A single comprehensive import cell at the beginning.
2.  Unique function/class definitions. If a function is *very* similar across TPs but with minor specific changes, I might create variants (e.g., `create_model_mlp_mnist`, `create_model_cnn_mnist`) or make a more generic function if feasible.
3.  For Pytorch models (which are classes), I'll use `## class ClassName(...)` in the markdown.
4.  The explanations for inputs/outputs will be derived from the context in the TP protocols and code.

Let's start building your consolidated notebook.

```python
# Execute this cell to ensure the notebook is in Python
```

```python
# ## Imports
# This cell contains all necessary imports aggregated from TP1 through TP7.
# Standard Libraries
import os
import urllib.request
import zipfile
import glob
import warnings

# Data Handling & Numerics
import numpy as np
import pandas as pd

# Plotting & Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2 # OpenCV

# Machine Learning - General
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score

# Machine Learning - Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# PyTorch Core
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, random_split

# Torchvision
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.v2 as transforms # For TP5 onwards, TP4 used torchvision.transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image # For XAI

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_fabric.utilities.seed import seed_everything as fabric_seed_everything # newer versions

# Torchinfo
from torchinfo import summary

# Torchmetrics
from torchmetrics import Accuracy # Older import style in TPs
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix # Newer/specific imports

# WandB
import wandb

# Folium (Mapping)
import folium
import branca.colormap

# PyCaret
from pycaret.regression import setup as pycaret_setup, compare_models as pycaret_compare_models, tune_model as pycaret_tune_model, evaluate_model as pycaret_evaluate_model, predict_model as pycaret_predict_model

# XAI Libraries (TP7)
from torchcam.methods import GradCAM, LayerCAM
from torchcam.utils import overlay_mask
from captum.attr import GradientShap, Occlusion, IntegratedGradients, DeepLift
from captum.attr import visualization as captum_viz
# For TIS (Transformer Input Sampling) - assuming it's in a subfolder or installed
# from Transformer_Input_Sampling.tis import TIS # This might require specific path setup
import timm
# import fast_pytorch_kmeans # If used by TIS
from skimage.transform import resize as skimage_resize # for RISE example potentially
from einops import rearrange # Often used with Transformers

# Pillow (PIL)
from PIL import Image

# Other utilities
from IPython.display import display

# Configuration for reproducibility and environment
# seed_everything(42, workers=True) # Ensure this is called once if needed globally
# torch.use_deterministic_algorithms(True) # If strict determinism is required
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # For CUDA determinism
warnings.filterwarnings('ignore', category=UserWarning, message='The given NumPy array is not writable')
```

```python
## def seed_everything_global(seed: int, workers: bool = True)
# Sets the seed for random number generators in PyTorch, NumPy, and Python's random module
# to ensure reproducibility. It also configures PyTorch to use deterministic algorithms
# if available and sets environment variables for CUDA determinism.

# Entrées :
# - seed (int): The seed value to use for all random number generators.
# - workers (bool): If True, sets the `workers` argument in `pytorch_lightning.seed_everything`.

# Sorties :
# - None
def seed_everything_global(seed: int, workers: bool = True):
    """
    Sets seeds for reproducibility across PyTorch, NumPy, and Python's random module.
    Also configures PyTorch for deterministic behavior.
    """
    # PyTorch Lightning's seed_everything also handles torch, numpy, and random.
    pl.seed_everything(seed, workers=workers)

    # For stricter determinism with CUDA (optional, might impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"
    print(f"Global seed set to {seed}")

# Example of calling it once at the beginning of a script
# seed_everything_global(42)
```

```python
## def create_simple_dataframe_pandas(data_dict: dict)
# Creates a Pandas DataFrame from a Python dictionary.
# This was used in TP1 for a basic introduction to Pandas.

# Entrées :
# - data_dict (dict): A dictionary where keys are column names and values are lists representing column data.

# Sorties :
# - df (pd.DataFrame): The Pandas DataFrame created from the dictionary.
import pandas as pd

def create_simple_dataframe_pandas(data_dict: dict) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from a Python dictionary.
    Example: data = {"col1": [1, 2], "col2": [3, 4]}
    """
    df = pd.DataFrame(data_dict)
    return df
```

```python
## def sort_dataframe_pandas(df: pd.DataFrame, by_columns: list, ascending_orders: list)
# Sorts a Pandas DataFrame by one or more columns, each with a specified sort order.
# Used in TP1 for data exploration.

# Entrées :
# - df (pd.DataFrame): The DataFrame to be sorted.
# - by_columns (list): A list of column names to sort by.
# - ascending_orders (list): A list of booleans corresponding to `by_columns`,
#                            where True means ascending and False means descending.

# Sorties :
# - df_sorted (pd.DataFrame): The sorted DataFrame.
import pandas as pd

def sort_dataframe_pandas(df: pd.DataFrame, by_columns: list, ascending_orders: list) -> pd.DataFrame:
    """
    Sorts a Pandas DataFrame by specified columns and orders.
    Example: sort_dataframe_pandas(df, by_columns=["population", "superficie_km2"], ascending_orders=[False, False])
    """
    if len(by_columns) != len(ascending_orders):
        raise ValueError("Length of by_columns and ascending_orders must be the same.")
    df_sorted = df.sort_values(by=by_columns, ascending=ascending_orders)
    return df_sorted
```

```python
## def plot_bar_seaborn(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, palette: str = "Blues_r", hue_col: str = None, figsize: tuple = (8, 5), orient: str = 'h')
# Creates a bar plot (horizontal or vertical) using Seaborn.
# Used in TP1 for visualizing city populations/areas.

# Entrées :
# - df (pd.DataFrame): The DataFrame containing the data.
# - x_col (str): The name of the column for the x-axis (or value-axis for horizontal).
# - y_col (str): The name of the column for the y-axis (or category-axis for horizontal).
# - title (str): The title of the plot.
# - xlabel (str): The label for the x-axis.
# - ylabel (str): The label for the y-axis.
# - palette (str, optional): Seaborn color palette. Defaults to "Blues_r".
# - hue_col (str, optional): Column name for color encoding. Defaults to None.
# - figsize (tuple, optional): Figure size. Defaults to (8, 5).
# - orient (str, optional): Orientation of the bar plot, 'h' for horizontal, 'v' for vertical. Defaults to 'h'.


# Sorties :
# - None (displays the plot).
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_bar_seaborn(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, palette: str = "Blues_r", hue_col: str = None, figsize: tuple = (8, 5), orient: str = 'h'):
    """
    Creates a bar plot using Seaborn.
    For horizontal bar plot (orient='h'): y_col is categorical, x_col is numerical.
    For vertical bar plot (orient='v'): x_col is categorical, y_col is numerical.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    if orient == 'h':
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette, hue=hue_col, orient='h')
    elif orient == 'v':
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette, hue=hue_col, orient='v')
    else:
        raise ValueError("orient must be 'h' or 'v'")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
```

```python
## def preprocess_housing_features_v1(california_housing_dataframe: pd.DataFrame)
# Preprocesses features for the California housing dataset (as in TP1/TP2).
# Selects specific features and creates a synthetic feature "rooms_per_person".

# Entrées :
# - california_housing_dataframe (pd.DataFrame): The input DataFrame with California housing data.
#   Expected columns: "latitude", "longitude", "housing_median_age", "total_rooms",
#   "total_bedrooms", "population", "households", "median_income".

# Sorties :
# - processed_features (pd.DataFrame): DataFrame with selected and engineered features.
import pandas as pd

def preprocess_housing_features_v1(california_housing_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Selects features and creates a synthetic feature "rooms_per_person"
    for the California housing dataset (TP1/TP2 version).
    """
    processed_features = california_housing_dataframe.copy()
    selected_features = ["latitude", "longitude", "housing_median_age", "total_rooms",
                         "total_bedrooms", "population", "households", "median_income"]
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]
    
    # Add the new feature to the list of selected features if it's not already implicitly included
    # and ensure only existing columns are selected if some original ones are missing.
    final_selected_features = selected_features + ["rooms_per_person"]
    
    # Filter out any feature names that might not exist in processed_features, though "rooms_per_person" was just added.
    # This primarily handles cases where input `california_housing_dataframe` might be missing some `selected_features`.
    existing_features_to_select = [col for col in final_selected_features if col in processed_features.columns]
    
    return processed_features[existing_features_to_select]
```

```python
## def preprocess_housing_targets_v1(california_housing_dataframe: pd.DataFrame)
# Preprocesses the target variable "median_house_value" for the California housing dataset (as in TP1/TP2).
# Scales the "median_house_value" by dividing by 1000.

# Entrées :
# - california_housing_dataframe (pd.DataFrame): The input DataFrame with California housing data.
#   Expected column: "median_house_value".

# Sorties :
# - processed_targets (pd.Series): Series containing the scaled "median_house_value".
import pandas as pd

def preprocess_housing_targets_v1(california_housing_dataframe: pd.DataFrame) -> pd.Series:
    """
    Scales the "median_house_value" by dividing by 1000.0 (TP1/TP2 version).
    Returns only the target column as a Series.
    """
    processed_data = california_housing_dataframe.copy() # Make a copy to avoid SettingWithCopyWarning
    processed_data["median_house_value"] = processed_data["median_house_value"] / 1000.0
    return processed_data["median_house_value"]
```

```python
## def plot_housing_data_scatter_map(training_examples: pd.DataFrame, validation_examples: pd.DataFrame)
# Visualizes training and validation data for California housing on a scatter plot,
# mimicking a map using longitude and latitude, colored by median_house_value.
# Used in TP1.

# Entrées :
# - training_examples (pd.DataFrame): DataFrame for training, must include "longitude", "latitude", "median_house_value".
# - validation_examples (pd.DataFrame): DataFrame for validation, must include "longitude", "latitude", "median_house_value".

# Sorties :
# - None (displays the plot).
import matplotlib.pyplot as plt
import pandas as pd

def plot_housing_data_scatter_map(training_examples: pd.DataFrame, validation_examples: pd.DataFrame):
  """
  Visualizes training and validation housing data on scatter plots.
  Longitude and latitude are used for x and y axes.
  Points are colored by median_house_value.
  """
  plt.figure(figsize=(13, 8))

  # Validation Data Plot
  ax1 = plt.subplot(1, 2, 1)
  ax1.set_title("Validation Data")
  ax1.set_autoscaley_on(False)
  ax1.set_ylim([32, 43])
  ax1.set_autoscalex_on(False)
  ax1.set_xlim([-126, -112])
  plt.scatter(validation_examples["longitude"],
              validation_examples["latitude"],
              cmap="coolwarm",
              c=validation_examples["median_house_value"] / validation_examples["median_house_value"].max())
  ax1.set_xlabel("Longitude")
  ax1.set_ylabel("Latitude")

  # Training Data Plot
  ax2 = plt.subplot(1,2,2)
  ax2.set_title("Training Data")
  ax2.set_autoscaley_on(False)
  ax2.set_ylim([32, 43])
  ax2.set_autoscalex_on(False)
  ax2.set_xlim([-126, -112])
  plt.scatter(training_examples["longitude"],
              training_examples["latitude"],
              cmap="coolwarm",
              c=training_examples["median_house_value"] / training_examples["median_house_value"].max())
  ax2.set_xlabel("Longitude")
  ax2.set_ylabel("Latitude")
  
  plt.tight_layout()
  plt.show()
```

```python
## def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson', figsize: tuple = (12, 8), annot: bool = True, fmt: str = '.2f', cmap: str = 'coolwarm')
# Calculates and displays a correlation heatmap for a DataFrame.
# Used in TP1.

# Entrées :
# - df (pd.DataFrame): The DataFrame for which to calculate correlations.
# - method (str, optional): Method of correlation ('pearson', 'kendall', 'spearman'). Defaults to 'pearson'.
# - figsize (tuple, optional): Figure size. Defaults to (12, 8).
# - annot (bool, optional): If True, write the data value in each cell. Defaults to True.
# - fmt (str, optional): String formatting code to use when `annot` is True. Defaults to '.2f'.
# - cmap (str, optional): Matplotlib colormap name or object. Defaults to 'coolwarm'.

# Sorties :
# - corr_matrix (pd.DataFrame): The calculated correlation matrix.
# - Displays the heatmap plot.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson', figsize: tuple = (12, 8), annot: bool = True, fmt: str = '.2f', cmap: str = 'coolwarm') -> pd.DataFrame:
    """
    Calculates and displays a correlation heatmap for the given DataFrame.
    """
    plt.figure(figsize=figsize)
    corr_matrix = df.corr(method=method)
    sns.heatmap(corr_matrix, annot=annot, fmt=fmt, cmap=cmap, center=0)
    plt.title(f"Correlation Matrix ({method.capitalize()})")
    plt.show()
    return corr_matrix
```

```python
## def plot_distribution_sns(series: pd.Series, title: str, xlabel: str, ylabel: str = "Frequency", bins: int = 30, kde: bool = True, figsize: tuple = (10, 6), color: str = 'blue')
# Displays the distribution of a numerical variable using Seaborn's histplot (distplot is deprecated).
# Used in TP1.

# Entrées :
# - series (pd.Series): The Pandas Series whose distribution is to be plotted.
# - title (str): The title of the plot.
# - xlabel (str): The label for the x-axis.
# - ylabel (str, optional): The label for the y-axis. Defaults to "Frequency".
# - bins (int, optional): Number of bins for the histogram. Defaults to 30.
# - kde (bool, optional): Whether to plot a kernel density estimate. Defaults to True.
# - figsize (tuple, optional): Figure size. Defaults to (10, 6).
# - color (str, optional): Color of the plot. Defaults to 'blue'.

# Sorties :
# - None (displays the plot).
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution_sns(series: pd.Series, title: str, xlabel: str, ylabel: str = "Frequency", bins: int = 30, kde: bool = True, figsize: tuple = (10, 6), color: str = 'blue'):
    """
    Displays the distribution of a numerical variable using Seaborn's histplot.
    """
    plt.figure(figsize=figsize)
    sns.histplot(series, kde=kde, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
```

```python
## def visualize_data_on_map_folium(data: pd.DataFrame, data_split_name: str, lat_col: str = 'latitude', lon_col: str = 'longitude', value_col: str = 'median_house_value', center_lat: float = 37.16, center_lon: float = -120.43, zoom: int = 6)
# Visualizes data points on a Folium map, colored by a specified value.
# Used in TP1.

# Entrées :
# - data (pd.DataFrame): DataFrame containing latitude, longitude, and the value to visualize.
# - data_split_name (str): Name of the data split (e.g., "Training Data", "Validation Data") for the title.
# - lat_col (str, optional): Name of the latitude column. Defaults to 'latitude'.
# - lon_col (str, optional): Name of the longitude column. Defaults to 'longitude'.
# - value_col (str, optional): Name of the column whose values will determine point color. Defaults to 'median_house_value'.
# - center_lat (float, optional): Latitude for the map center. Defaults to 37.16.
# - center_lon (float, optional): Longitude for the map center. Defaults to -120.43.
# - zoom (int, optional): Initial zoom level for the map. Defaults to 6.


# Sorties :
# - m (folium.Map): The Folium map object.
import pandas as pd
import folium
import branca.colormap

def visualize_data_on_map_folium(data: pd.DataFrame, data_split_name: str, lat_col: str = 'latitude', lon_col: str = 'longitude', value_col: str = 'median_house_value', center_lat: float = 37.16, center_lon: float = -120.43, zoom: int = 6) -> folium.Map:
    """
    Visualizes data points on a Folium map, colored by a specified value.
    """
    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_start=zoom)

    # Define a colormap based on the median house value
    # Ensure value_col exists and is numeric
    if value_col not in data.columns or not pd.api.types.is_numeric_dtype(data[value_col]):
        print(f"Warning: Column '{value_col}' not found or not numeric. Using default radius/color.")
        use_colormap = False
    else:
        use_colormap = True
        min_val = data[value_col].min()
        max_val = data[value_col].max()
        if min_val == max_val: # Avoid division by zero if all values are the same
             colormap = branca.colormap.LinearColormap(colors=['blue', 'blue'], index=[0,1], vmin=0, vmax=1)
        else:
            colormap = branca.colormap.LinearColormap(colors=['blue', 'red'], index=[min_val, max_val], vmin=min_val, vmax=max_val)


    for i in range(len(data)):
        point_color = 'blue' # Default color
        fill_opacity = 0.7
        if use_colormap:
            current_value = data.iloc[i][value_col]
            if min_val == max_val:
                 normalized_value = 0.5 # or any fixed value
            else:
                normalized_value = (current_value - min_val) / (max_val - min_val)
            point_color = colormap(current_value)


        folium.Circle(
            location=[data.iloc[i][lat_col], data.iloc[i][lon_col]],
            radius=50, # Adjust radius as needed, can also be data-driven
            color=point_color,
            fill=True,
            fill_color=point_colorm,
            fill_opacity=fill_opacity
        ).add_to(m)

    title_html = f'<h3 align="center" style="font-size:16px"><b>{data_split_name}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    if use_colormap: # Add colormap legend only if used
        m.add_child(colormap)
    return m
```

```python
## class SimpleLinearModelPytorch(nn.Module)
# A simple linear regression model (y = ax + b) using PyTorch's nn.Module.
# Used in TP2 for an introduction to PyTorch model definition.

# Entrées (pour __init__) :
# - input_dim (int): Dimensionality of the input features (typically 1 for simple linear regression).
# - output_dim (int): Dimensionality of the output (typically 1 for simple linear regression).

# Entrées (pour forward) :
# - x (torch.Tensor): The input tensor.

# Sorties (de forward) :
# - (torch.Tensor): The output tensor (predictions).
import torch
import torch.nn as nn

class SimpleLinearModelPytorch(nn.Module):
    """
    A simple linear regression model (y = ax + b) using PyTorch.
    input_dim: number of input features (e.g., 1 for X)
    output_dim: number of output values (e.g., 1 for Y)
    """
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super(SimpleLinearModelPytorch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

```python
## def train_simple_linear_model_pytorch(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, epochs: int = 100, print_every: int = 10, plot_every: int = 0)
# Trains a simple PyTorch model (like SimpleLinearModelPytorch).
# Handles the training loop: forward pass, loss calculation, backward pass, optimizer step.
# Used in TP2.

# Entrées :
# - model (nn.Module): The PyTorch model to train.
# - criterion (nn.Module): The loss function (e.g., nn.MSELoss()).
# - optimizer (optim.Optimizer): The optimizer (e.g., torch.optim.SGD(...)).
# - X_tensor (torch.Tensor): The input data tensor.
# - Y_tensor (torch.Tensor): The target data tensor.
# - epochs (int, optional): Number of training epochs. Defaults to 100.
# - print_every (int, optional): Frequency of printing loss (e.g., every 10 epochs). If 0, no printing.
# - plot_every (int, optional): Frequency of plotting predictions vs original. If 0, no plotting during training.

# Sorties :
# - losses (list): A list of loss values, one for each epoch.
# - Displays loss information and optional plots during training.
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_simple_linear_model_pytorch(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, epochs: int = 100, print_every: int = 10, plot_every: int = 0) -> list:
    """
    Trains a simple PyTorch model.
    `plot_every > 0` will plot predictions assuming X_tensor and Y_tensor are 1D for scatter plot.
    """
    losses = []
    
    # Ensure X_tensor and Y_tensor are float tensors
    if not isinstance(X_tensor, torch.FloatTensor):
        X_tensor = X_tensor.float()
    if not isinstance(Y_tensor, torch.FloatTensor):
        Y_tensor = Y_tensor.float()

    # Reshape if they are 1D for nn.Linear compatibility
    if X_tensor.ndim == 1:
        X_tensor = X_tensor.view(-1, 1)
    if Y_tensor.ndim == 1:
        Y_tensor = Y_tensor.view(-1, 1)

    for epoch in range(epochs):
        # 1. Reset gradients
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(X_tensor)

        # 3. Calculate loss
        loss = criterion(predictions, Y_tensor)
        losses.append(loss.item())

        # 4. Backward pass
        loss.backward()

        # 5. Update weights
        optimizer.step()

        if print_every > 0 and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        if plot_every > 0 and (epoch + 1) % plot_every == 0 :
            plt.figure(figsize=(6, 4))
            # Detach tensors and convert to numpy for plotting
            # Assuming X_tensor was originally 1D for this specific plot
            original_X_for_plot = X_tensor.view(-1).detach().cpu().numpy()
            original_Y_for_plot = Y_tensor.view(-1).detach().cpu().numpy()
            predicted_Y_for_plot = predictions.view(-1).detach().cpu().numpy()

            plt.scatter(original_X_for_plot, original_Y_for_plot, color='blue', label='Original Data', s=10)
            plt.scatter(original_X_for_plot, predicted_Y_for_plot, color='red', label='Predictions', s=10, alpha=0.6)
            plt.plot(original_X_for_plot, predicted_Y_for_plot, color='red', linestyle='--', alpha=0.5) # Line for trend
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Epoch {epoch+1} Predictions')
            plt.legend()
            plt.show()
            
    return losses
```

```python
## def scale_data_minmax(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None, X_val: pd.DataFrame = None, y_val: pd.Series = None)
# Scales features (X) and target (y) using MinMaxScaler.
# Fits scaler on training data and transforms train, validation, and test sets.
# Target variable `y` is reshaped to 2D before scaling, as MinMaxScaler expects 2D input.
# Used in TP2 for housing data and TP3 (adapted as normalize_data).

# Entrées :
# - X_train (pd.DataFrame): Training features.
# - y_train (pd.Series): Training target.
# - X_test (pd.DataFrame, optional): Test features.
# - y_test (pd.Series, optional): Test target.
# - X_val (pd.DataFrame, optional): Validation features.
# - y_val (pd.Series, optional): Validation target.

# Sorties :
# - tuple: Contains scaled data (X_train_scaled, y_train_scaled, ...),
#          and the fitted scalers (scaler_X, scaler_y).
#          Order: X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, scaler_X, scaler_y
#          If val/test data are not provided, corresponding outputs will be None.
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def scale_data_minmax(X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame = None, y_test: pd.Series = None, 
                      X_val: pd.DataFrame = None, y_val: pd.Series = None):
    """
    Scales features (X) and target (y) using MinMaxScaler.
    Fits on training data and transforms train, val, and test sets.
    y is reshaped to 2D for scaler compatibility.
    Returns scaled data and fitted scalers.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale features
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler_X.transform(X_test) if X_test is not None else None

    # Reshape y to 2D array as MinMaxScaler expects 2D input
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped)

    y_val_scaled = None
    if y_val is not None:
        y_val_reshaped = y_val.values.reshape(-1, 1)
        y_val_scaled = scaler_y.transform(y_val_reshaped)

    y_test_scaled = None
    if y_test is not None:
        y_test_reshaped = y_test.values.reshape(-1, 1)
        y_test_scaled = scaler_y.transform(y_test_reshaped)

    # Convert scaled X back to DataFrames if original were DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    if X_val is not None:
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    if X_test is not None:
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
    # y_scaled are numpy arrays, which is typical for model input/output

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y
```

```python
## def calculate_regression_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, y_scaler: MinMaxScaler, epsilon: float = 1e-7)
# Calculates MAE, MAPE, and R2 score for regression tasks.
# Importantly, it first inverse-transforms the scaled true and predicted values
# using the provided y_scaler before calculating metrics, so metrics are in original scale.
# Used in TP2.

# Entrées :
# - y_true_scaled (np.ndarray): Scaled true target values (typically 2D, e.g., from MinMaxScaler).
# - y_pred_scaled (np.ndarray): Scaled predicted target values (typically 2D).
# - y_scaler (MinMaxScaler): The scaler object that was used to scale `y_true_scaled` (and `y_pred_scaled` implicitly).
# - epsilon (float, optional): Small value to add to denominator for MAPE to avoid division by zero.

# Sorties :
# - dict: A dictionary containing 'mae', 'mape', and 'r2'.
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def calculate_regression_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, y_scaler: MinMaxScaler, epsilon: float = 1e-7) -> dict:
    """
    Calculates MAE, MAPE, and R2 score after inverse-transforming scaled predictions.
    Assumes y_true_scaled and y_pred_scaled are 2D numpy arrays (output of MinMaxScaler).
    """
    # Ensure inputs are numpy arrays and reshape if they are 1D (e.g. Series.values)
    y_true_s = np.asarray(y_true_scaled)
    y_pred_s = np.asarray(y_pred_scaled)

    if y_true_s.ndim == 1:
        y_true_s = y_true_s.reshape(-1, 1)
    if y_pred_s.ndim == 1:
        y_pred_s = y_pred_s.reshape(-1, 1)

    # Inverse transform to original scale
    y_true_unscaled = y_scaler.inverse_transform(y_true_s)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_s)

    # Calculate MAE
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    # Calculate MAPE
    # Ensure y_true_unscaled is not zero for division
    # Use np.maximum to avoid division by zero or very small numbers
    diff = np.abs(y_true_unscaled - y_pred_unscaled)
    denominator = np.maximum(np.abs(y_true_unscaled), epsilon) 
    mape = np.mean(diff / denominator) * 100 # As percentage

    # Calculate R2 Score
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)

    return {"mae": mae, "mape": mape, "r2": r2}
```

```python
## def plot_regression_predictions(y_true_original_scale: pd.Series, y_pred_original_scale: np.ndarray, num_samples_to_plot: int = 100, title: str = "Predictions vs Real Values", xlabel: str = "House Index", ylabel: str = "House Value")
# Plots true vs. predicted values for a regression model.
# Assumes inputs are already in their original, unscaled form.
# Used in TP2.

# Entrées :
# - y_true_original_scale (pd.Series or np.ndarray): True target values in original scale.
# - y_pred_original_scale (np.ndarray): Predicted target values in original scale.
# - num_samples_to_plot (int, optional): Number of initial samples to plot. Defaults to 100.
# - title (str, optional): Title of the plot.
# - xlabel (str, optional): Label for the x-axis.
# - ylabel (str, optional): Label for the y-axis.

# Sorties :
# - None (displays the plot).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_predictions(y_true_original_scale: pd.Series, y_pred_original_scale: np.ndarray, num_samples_to_plot: int = 100, title: str = "Predictions vs Real Values", xlabel: str = "House Index", ylabel: str = "House Value"):
    """
    Plots true vs. predicted values for a regression model.
    Inputs y_true and y_pred should be in their original (unscaled) form.
    y_true_original_scale can be a Pandas Series or a 1D NumPy array.
    y_pred_original_scale should be a 1D NumPy array.
    """
    if isinstance(y_true_original_scale, pd.Series):
        y_true_plot = y_true_original_scale.values[:num_samples_to_plot]
    else:
        y_true_plot = np.asarray(y_true_original_scale)[:num_samples_to_plot]
        
    y_pred_plot = np.asarray(y_pred_original_scale).flatten()[:num_samples_to_plot] # Ensure 1D

    result_df = pd.DataFrame({
        'Index': list(range(min(len(y_true_plot), len(y_pred_plot)))), # Ensure index matches shortest array
        'Real Values': y_true_plot,
        'Predictions': y_pred_plot
    })

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=result_df, x='Index', y='Real Values', marker='o', label='Real Values', linestyle='-')
    sns.lineplot(data=result_df, x='Index', y='Predictions', marker='x', label='Predictions', linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
```

```python
## def preprocess_housing_features_v2_age(housing_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
# Preprocesses features and extracts target for a different housing dataset (KC Housing Data - TP3).
# Creates an 'age' feature from 'sales_yr' (derived from 'date') and 'yr_built'.
# Selects a specific list of features.

# Entrées :
# - housing_dataframe (pd.DataFrame): The input DataFrame (e.g., kc_house_data.csv).
#   Expected columns: 'date', 'yr_built', and other features listed in `selected_features`.
#   Target column: 'price'.

# Sorties :
# - X (pd.DataFrame): DataFrame with selected and engineered features.
# - Y (pd.Series): Series containing the target 'price'.
import pandas as pd

def preprocess_housing_features_v2_age(housing_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses features for the KC housing dataset (TP3).
    Adds 'age' column, selects features, and separates target 'price'.
    """
    housing_copy = housing_dataframe.copy()
    
    # Ensure 'date' column is in datetime format
    if 'date' in housing_copy.columns:
        housing_copy['sales_yr'] = pd.to_datetime(housing_copy['date']).dt.year
        # Add the age of the buildings when the houses were sold as a new column
        if 'yr_built' in housing_copy.columns:
            housing_copy['age'] = housing_copy['sales_yr'] - housing_copy['yr_built']
        else:
            raise KeyError("'yr_built' column not found, cannot calculate 'age'.")
    else:
        # If 'date' is not present, 'age' cannot be calculated this way.
        # Depending on requirements, either raise error or skip 'age' calculation.
        # For now, we assume 'age' might be pre-calculated or comes from elsewhere if 'date' is missing.
        if 'age' not in housing_copy.columns:
            print("Warning: 'date' column not found and 'age' not pre-existing. 'age' feature will be missing.")
            # housing_copy['age'] = 0 # Or some default if absolutely needed by selected_features

    selected_features_list = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 
        'long', 'sqft_living15', 'sqft_lot15'
    ]
    if 'age' in housing_copy.columns: # Only add 'age' if it was successfully created
        selected_features_list.append('age')

    # Filter selected_features_list to only include columns that exist in housing_copy
    final_selected_features = [col for col in selected_features_list if col in housing_copy.columns]
    
    X = housing_copy[final_selected_features]
    
    if 'price' in housing_copy.columns:
        Y = housing_copy['price']
    else:
        raise KeyError("'price' column (target) not found in the DataFrame.")
        
    return X, Y
```

```python
## class TabularDataset(Dataset)
# A PyTorch Dataset class for tabular data.
# Converts Pandas DataFrame/Series slices into PyTorch tensors.
# Used in TP3.

# Entrées (__init__) :
# - data (np.ndarray or pd.DataFrame): The feature data.
# - labels (np.ndarray or pd.Series): The label data.

# Entrées (__getitem__) :
# - idx (int): Index of the data sample to retrieve.

# Sorties (__getitem__) :
# - x (torch.Tensor): Tensor of features for the sample.
# - y (torch.Tensor): Tensor of labels for the sample.
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    Assumes data and labels are NumPy arrays or can be converted.
    """
    def __init__(self, data, labels):
        # Convert to numpy array if pandas DataFrame/Series
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(np.float32)
        else:
            self.data = np.asarray(data, dtype=np.float32)
            
        if isinstance(labels, pd.Series):
            self.labels = labels.values.astype(np.float32)
        else:
            self.labels = np.asarray(labels, dtype=np.float32)

        # Ensure labels are 2D if they are not already (e.g. for regression tasks with MSELoss)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32) # y should also be float for most regression losses
        return x, y
```

```python
## def calculate_regression_metrics_pytorch(y_true_tensor: torch.Tensor, y_pred_tensor: torch.Tensor, y_scaler: MinMaxScaler) -> dict
# Calculates MAE and R2 score for regression tasks using PyTorch tensors.
# Converts tensors to NumPy arrays and inverse-transforms using y_scaler before metric calculation.
# Used in TP3 (BaseModel).

# Entrées :
# - y_true_tensor (torch.Tensor): True target values (scaled).
# - y_pred_tensor (torch.Tensor): Predicted target values (scaled).
# - y_scaler (MinMaxScaler): The scaler object for inverse transformation.

# Sorties :
# - dict: A dictionary containing 'mae' and 'r2'.
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def calculate_regression_metrics_pytorch(y_true_tensor: torch.Tensor, y_pred_tensor: torch.Tensor, y_scaler: MinMaxScaler) -> dict:
    """
    Calculates MAE and R2 score from PyTorch tensors after inverse scaling.
    """
    # Detach tensors from graph, move to CPU, convert to NumPy
    y_true_np = y_true_tensor.detach().cpu().numpy()
    y_pred_np = y_pred_tensor.detach().cpu().numpy()

    # Reshape if they are not 2D (MinMaxScaler expects 2D)
    if y_true_np.ndim == 1:
        y_true_np = y_true_np.reshape(-1, 1)
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np.reshape(-1, 1)

    # Inverse transform
    y_true_unscaled = y_scaler.inverse_transform(y_true_np)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_np)

    # Calculate metrics
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)

    return {"mae": mae, "r2": r2}
```

```python
## class BaseLightningModel(pl.LightningModule)
# A base PyTorch Lightning module for regression tasks.
# Implements training_step, validation_step, test_step, predict_step, and configure_optimizers.
# Logs MAE and R2 metrics.
# Used in TP3.

# Entrées (__init__) :
# - model (nn.Module): The PyTorch neural network model.
# - loss_fn (nn.Module): The loss function (e.g., nn.MSELoss()).
# - y_scaler (MinMaxScaler): Scaler for the target variable, used for unscaling metrics.
# - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 6e-3.
# - optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use. Defaults to torch.optim.Adam.


# Sorties :
# - (Implicitly, during training/testing) Logs metrics to the logger.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, r2_score # Already imported or use custom

class BaseLightningModel(pl.LightningModule):
    """
    Base PyTorch Lightning module for regression tasks.
    Logs MAE and R2 metrics, using the provided y_scaler to report them in original scale.
    """
    def __init__(self, model: nn.Module, loss_fn: nn.Module, y_scaler: MinMaxScaler, learning_rate: float = 6e-3, optimizer_class = torch.optim.Adam):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.y_scaler = y_scaler # For unscaling metrics
        self.lr = learning_rate
        self.optimizer_class = optimizer_class
        # self.save_hyperparameters() # Can be useful for PTL checkpointing, excluding model, loss_fn, y_scaler if they are not simple types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage: str):
        x, y_true_scaled = batch
        y_pred_scaled = self(x)
        loss = self.loss_fn(y_pred_scaled, y_true_scaled)
        
        # Calculate metrics on unscaled data
        metrics_unscaled = calculate_regression_metrics_pytorch(y_true_scaled, y_pred_scaled, self.y_scaler)
        
        log_dict = {
            f"{stage}_loss": loss,
            f"{stage}_mae_unscaled": metrics_unscaled["mae"],
            f"{stage}_r2_unscaled": metrics_unscaled["r2"]
        }
        # prog_bar=True makes them appear in the progress bar
        self.log_dict(log_dict, on_step=(stage=="train"), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Batch might be just x, or (x, y)
        if isinstance(batch, tuple) or isinstance(batch, list):
            x, _ = batch # if y is present, ignore it for prediction
        else:
            x = batch
        y_pred_scaled = self(x)
        # Inverse transform predictions to original scale
        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred_scaled.detach().cpu().numpy())
        return torch.tensor(y_pred_unscaled, dtype=torch.float32) # Return as tensor

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
```

```python
## class MLPModel(nn.Module)
# Generic Multi-Layer Perceptron (MLP) model.
# Can create MLPs with a variable number of layers, hidden units, activation functions, and dropout.
# Used for TP3 (housing prediction) and adaptable for TP4 (MNIST) if input_dim/output_dim are set.

# Entrées (__init__) :
# - input_dim (int): Dimensionality of the input layer.
# - output_dim (int): Dimensionality of the output layer.
# - hidden_layers (list of int, optional): List where each int is the number of neurons in a hidden layer.
#   Defaults to [64, 32] (two hidden layers).
# - activation_fn (nn.Module, optional): Activation function to use after each hidden layer.
#   Defaults to nn.ReLU(). If None, no activation is applied (linear layers).
# - dropout_rates (list of float, optional): List of dropout rates to apply after each hidden layer's activation.
#   Length must match `hidden_layers`. If None or empty, no dropout. Defaults to None.
# - final_activation_fn (nn.Module, optional): Activation function for the output layer. Defaults to None.


# Entrées (forward) :
# - x (torch.Tensor): The input tensor. If model expects flattened input (e.g., for images),
#                     flattening should happen before this call or as the first step here.

# Sorties (de forward) :
# - (torch.Tensor): The output tensor (predictions).
import torch
import torch.nn as nn
from collections import OrderedDict

class MLPModel(nn.Module):
    """
    Generic Multi-Layer Perceptron (MLP) model.
    Allows specifying number of hidden layers, units, activation, and dropout.
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_layers: list = [64, 32], 
                 activation_fn: nn.Module = nn.ReLU(),
                 dropout_rates: list = None,
                 final_activation_fn: nn.Module = None):
        super(MLPModel, self).__init__()
        
        layers = OrderedDict()
        current_dim = input_dim

        for i, h_dim in enumerate(hidden_layers):
            layers[f'fc{i+1}'] = nn.Linear(current_dim, h_dim)
            if activation_fn is not None:
                layers[f'act{i+1}'] = activation_fn
            if dropout_rates and i < len(dropout_rates) and dropout_rates[i] > 0:
                layers[f'dropout{i+1}'] = nn.Dropout(dropout_rates[i])
            current_dim = h_dim
        
        layers['output_fc'] = nn.Linear(current_dim, output_dim)
        if final_activation_fn is not None:
            layers['output_act'] = final_activation_fn
            
        self.network = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x is an image batch (e.g., [batch_size, C, H, W]), flatten it.
        # This MLP expects 1D features per sample.
        if x.ndim > 2: # Assumes input_dim was calculated based on flattened size
            x = x.view(x.size(0), -1) 
        return self.network(x)

# Example Usage for TP3 Models:
# model_single_layer_tp3 = MLPModel(input_dim=19, output_dim=1, hidden_layers=[], activation_fn=None) # Linear layer
# model_2_layers_tp3 = MLPModel(input_dim=19, output_dim=1, hidden_layers=[64], activation_fn=nn.ReLU())
# model_4_layers_tp3 = MLPModel(input_dim=19, output_dim=1, hidden_layers=[128, 64, 32], activation_fn=nn.ReLU())
# model_5_layers_dropout_tp3 = MLPModel(input_dim=19, output_dim=1, 
#                                      hidden_layers=[256, 128, 64, 32], 
#                                      activation_fn=nn.ReLU(),
#                                      dropout_rates=[0.2, 0.2, 0.0, 0.0]) # Dropout after first two ReLUs
```

```python
## def plot_regression_predictions_pytorch(predictions_list: list, y_true_series: pd.Series, y_scaler: MinMaxScaler, num_samples_to_plot: int = 100, title: str = "Predictions vs Real Values (PyTorch)", xlabel: str = "House Index", ylabel: str = "House Value")
# Plots true vs. predicted values for a PyTorch regression model.
# `predictions_list` is a list of batches of predicted tensors (output from trainer.predict).
# These predictions are concatenated, converted to NumPy, and inverse-transformed using y_scaler.
# `y_true_series` contains the original, unscaled true values.
# Used in TP3.

# Entrées :
# - predictions_list (list): List of PyTorch tensors, where each tensor is a batch of scaled predictions.
# - y_true_series (pd.Series): Pandas Series of true target values in their original (unscaled) scale.
# - y_scaler (MinMaxScaler): The scaler object used for the target variable, for inverse transformation of predictions.
# - num_samples_to_plot (int, optional): Number of initial samples to plot. Defaults to 100.
# - title (str, optional): Title of the plot.
# - xlabel (str, optional): Label for the x-axis.
# - ylabel (str, optional): Label for the y-axis.

# Sorties :
# - None (displays the plot).
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def plot_regression_predictions_pytorch(predictions_list: list, y_true_series: pd.Series, y_scaler: MinMaxScaler, num_samples_to_plot: int = 100, title: str = "Predictions vs Real Values (PyTorch)", xlabel: str = "House Index", ylabel: str = "House Value"):
    """
    Plots true vs. predicted values for PyTorch regression models.
    `predictions_list` contains batches of SCALED predictions from trainer.predict().
    `y_true_series` contains ORIGINAL (unscaled) true values.
    Predictions are inverse-transformed using y_scaler.
    """
    # Concatenate all prediction tensors from the list of batches
    if not predictions_list:
        print("Warning: predictions_list is empty. Nothing to plot.")
        return
        
    all_preds_scaled_tensor = torch.cat(predictions_list, dim=0)
    
    # Convert to NumPy and ensure it's 2D for the scaler
    all_preds_scaled_np = all_preds_scaled_tensor.detach().cpu().numpy()
    if all_preds_scaled_np.ndim == 1:
        all_preds_scaled_np = all_preds_scaled_np.reshape(-1, 1)

    # Inverse transform predictions
    y_pred_original_scale = y_scaler.inverse_transform(all_preds_scaled_np).flatten()

    # Get true values for plotting
    y_true_plot = y_true_series.values[:num_samples_to_plot]
    y_pred_plot = y_pred_original_scale[:num_samples_to_plot]
    
    # Ensure lengths match for DataFrame creation (in case y_true_series has fewer than num_samples_to_plot elements)
    actual_num_samples = min(len(y_true_plot), len(y_pred_plot))

    result_df = pd.DataFrame({
        'Index': list(range(actual_num_samples)),
        'Real Values': y_true_plot[:actual_num_samples],
        'Predictions': y_pred_plot[:actual_num_samples]
    })

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=result_df, x='Index', y='Real Values', marker='o', label='Real Values', linestyle='-')
    sns.lineplot(data=result_df, x='Index', y='Predictions', marker='x', label='Predictions', linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
```

```python
## class MNISTLightningModel(pl.LightningModule)
# PyTorch Lightning module for MNIST classification.
# Handles training, validation, and test steps, logging loss and accuracy.
# Used in TP4.

# Entrées (__init__) :
# - model (nn.Module): The PyTorch neural network (e.g., an MLP or CNN).
# - optimizer_class (torch.optim.Optimizer): The optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
# - num_classes (int, optional): Number of output classes (10 for MNIST). Defaults to 10.
# - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
# - weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 1e-4.

# Sorties :
# - (Implicitly) Logs metrics, manages training.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchmetrics import Accuracy # Older import
from torchmetrics.classification import MulticlassAccuracy # More specific

class MNISTLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for MNIST classification.
    """
    def __init__(self, model: nn.Module, optimizer_class, num_classes: int = 10, learning_rate: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.optimizer_class = optimizer_class
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        # Using MulticlassAccuracy
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        self.save_hyperparameters(ignore=['model']) # PTL saves hyperparameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage: str):
        x, y_true = batch
        logits = self(x) # Raw model output
        loss = F.cross_entropy(logits, y_true) # CrossEntropyLoss expects raw logits
        
        # For accuracy, need predicted class indices
        # preds = torch.argmax(logits, dim=1) # Not needed if metric handles logits
        
        acc_metric = getattr(self, f"{stage}_acc")
        acc = acc_metric(logits, y_true) # Pass logits directly
        
        self.log_dict({f'{stage}_loss': loss, f"{stage}_acc": acc}, 
                      on_step=(stage=="train"), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    # Optional: Reset metrics at epoch end if they accumulate state in a way not desired
    # Torchmetrics typically handles this well with on_epoch=True logging.
    # def on_train_epoch_end(self):
    #     self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    # def on_validation_epoch_end(self):
    #     self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    # def on_test_epoch_end(self):
    #     self.test_acc.reset()

    def configure_optimizers(self):
        # Note: self.model.parameters() is used if model is passed, 
        # or self.parameters() if the nn.Modules are direct attributes of this LightningModule
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
```

```python
## class CNNModel MNIST(nn.Module)
# Generic Convolutional Neural Network (CNN) model structure, adaptable for MNIST.
# Allows defining a sequence of convolutional blocks (Conv2d, Activation, MaxPool, Dropout)
# followed by a sequence of fully connected layers.
# Used for TP4.

# Entrées (__init__) :
# - input_channels (int): Number of channels in the input image (1 for MNIST grayscale).
# - num_classes (int): Number of output classes (10 for MNIST).
# - conv_layers_config (list of dicts): Configuration for convolutional layers.
#   Each dict: {'out_channels': int, 'kernel_size': int or tuple, 'stride': int or tuple, 'padding': int or tuple,
#               'pool_kernel_size': int or tuple (optional), 'dropout_rate': float (optional)}
# - fc_layers_config (list of dicts): Configuration for fully connected layers.
#   Each dict: {'out_features': int, 'dropout_rate': float (optional)}
# - activation_fn (nn.Module, optional): Activation function for conv and FC layers. Defaults to nn.ReLU().
# - initial_image_size (tuple, optional): (Height, Width) of the input image, needed if fc_layers_config
#                                         doesn't specify `in_features` for the first FC layer. Defaults to (28,28) for MNIST.

# Entrées (forward) :
# - x (torch.Tensor): Input tensor of shape (batch_size, input_channels, H, W).

# Sorties (de forward) :
# - (torch.Tensor): Output tensor of logits (batch_size, num_classes).

import torch
import torch.nn as nn
from collections import OrderedDict

class CNNModelMNIST(nn.Module):
    def __init__(self, input_channels: int, num_classes: int,
                 conv_layers_config: list,
                 fc_layers_config: list,
                 activation_fn: nn.Module = nn.ReLU(),
                 initial_image_size: tuple = (28, 28)):
        super(CNNModelMNIST, self).__init__()

        conv_modules = OrderedDict()
        current_channels = input_channels
        current_h, current_w = initial_image_size

        for i, cfg in enumerate(conv_layers_config):
            conv_modules[f'conv{i+1}'] = nn.Conv2d(
                in_channels=current_channels,
                out_channels=cfg['out_channels'],
                kernel_size=cfg['kernel_size'],
                stride=cfg.get('stride', 1),
                padding=cfg.get('padding', 0)
            )
            current_h = (current_h + 2 * cfg.get('padding', 0) - cfg['kernel_size']) // cfg.get('stride', 1) + 1
            current_w = (current_w + 2 * cfg.get('padding', 0) - cfg['kernel_size']) // cfg.get('stride', 1) + 1
            
            if activation_fn:
                conv_modules[f'act_conv{i+1}'] = activation_fn
            
            if 'pool_kernel_size' in cfg and cfg['pool_kernel_size']:
                conv_modules[f'pool{i+1}'] = nn.MaxPool2d(kernel_size=cfg['pool_kernel_size'], stride=cfg.get('pool_stride', cfg['pool_kernel_size']))
                current_h = current_h // cfg.get('pool_stride', cfg['pool_kernel_size'])
                current_w = current_w // cfg.get('pool_stride', cfg['pool_kernel_size'])

            if 'dropout_rate' in cfg and cfg['dropout_rate'] > 0:
                conv_modules[f'dropout_conv{i+1}'] = nn.Dropout2d(cfg['dropout_rate']) # Dropout2d for conv layers
            current_channels = cfg['out_channels']

        self.conv_layers = nn.Sequential(conv_modules)
        
        # Calculate the flattened size after conv layers
        # Use a dummy tensor to find the output shape of conv_layers
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, initial_image_size[0], initial_image_size[1])
            conv_output_shape = self.conv_layers(dummy_input).shape
            flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        fc_modules = OrderedDict()
        current_features = flattened_size
        for i, cfg in enumerate(fc_layers_config):
            fc_modules[f'fc{i+1}'] = nn.Linear(current_features, cfg['out_features'])
            if activation_fn and i < len(fc_layers_config) -1 : # No activation usually on last FC before softmax/crossentropy
                 fc_modules[f'act_fc{i+1}'] = activation_fn

            if 'dropout_rate' in cfg and cfg['dropout_rate'] > 0 and i < len(fc_layers_config) -1: # No dropout usually on last FC
                fc_modules[f'dropout_fc{i+1}'] = nn.Dropout(cfg['dropout_rate'])
            current_features = cfg['out_features']
        
        # Ensure last layer outputs num_classes
        # If fc_layers_config is empty or last layer doesn't match num_classes, add/replace last layer
        if not fc_layers_config or current_features != num_classes :
             fc_modules[f'fc_final_output'] = nn.Linear(current_features,num_classes)


        self.fc_layers = nn.Sequential(fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc_layers(x)
        return x

# Example Usage for TP4 CNNs:
# cnn_one_conv_config = [
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'pool_kernel_size': 2} # Valid padding (no padding)
# ]
# fc_one_conv_config = [
#     {'out_features': 128},
#     # Final layer to num_classes is implicitly handled by the class if not specified
# ]
# cnn1_tp4 = CNNModelMNIST(input_channels=1, num_classes=10, conv_layers_config=cnn_one_conv_config, fc_layers_config=fc_one_conv_config, initial_image_size=(28,28))
# print(summary(cnn1_tp4, input_size=(1, 1, 28, 28)))


# cnn_two_conv_config = [
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'dropout_rate': 0.2}, # Valid padding means kernel_size=3 reduces dim by 2
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'dropout_rate': 0.2, 'pool_kernel_size': 2}
# ]
# fc_two_conv_config = [
#     {'out_features': 128, 'dropout_rate': 0.2},
# ]
# cnn2_tp4 = CNNModelMNIST(input_channels=1, num_classes=10, conv_layers_config=cnn_two_conv_config, fc_layers_config=fc_two_conv_config, initial_image_size=(28,28))
# print(summary(cnn2_tp4, input_size=(1, 1, 28, 28)))
```

```python
## def create_image_classification_dataloaders(dataset_path: str, batch_size: int, train_split_ratio: float, val_split_ratio: float, img_size: int = 224, num_workers: int = 2, pin_memory: bool = True)
# Creates train, validation, and test DataLoaders for image classification from a directory structure
# (e.g., dataset_path/class_a/image1.jpg, dataset_path/class_b/image2.jpg).
# Applies standard transformations for training (augmentation) and evaluation.
# The dataset is split into train, validation, and test sets.
# Used in TP5, TP6, TP7.

# Entrées :
# - dataset_path (str): Path to the root directory of the dataset.
# - batch_size (int): Number of samples per batch.
# - train_split_ratio (float): Proportion of the dataset to allocate for training (e.g., 0.8 for 80%).
# - val_split_ratio (float): Proportion of the dataset to allocate for validation (e.g., 0.1 for 10%).
#                            The test set will be the remainder (1 - train_split_ratio - val_split_ratio).
# - img_size (int, optional): Size to resize images to (img_size x img_size). Defaults to 224.
# - num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 2.
# - pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.


# Sorties :
# - train_loader (DataLoader): DataLoader for the training set.
# - val_loader (DataLoader): DataLoader for the validation set.
# - test_loader (DataLoader): DataLoader for the test set.
# - class_names (list): List of class names found in the dataset.
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms # Using v2 for modern augmentations
import numpy as np

def create_image_classification_dataloaders(
    dataset_path: str, 
    batch_size: int, 
    train_split_ratio: float, 
    val_split_ratio: float, 
    img_size: int = 224,
    num_workers: int = 2, # os.cpu_count() can be a good default
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Creates train, validation, and test DataLoaders for image classification.
    Splits the dataset based on provided ratios.
    Applies different transformations for train (with augmentation) and eval sets.
    """

    # Transformations
    # Note: ToTensorV2() is part of transforms.v2, normal ToTensor is in original transforms
    # Normalization values are typical for ImageNet pre-trained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.PILToTensor(), # Converts PIL Image to Tensor
        transforms.ToDtype(torch.float32, scale=True), # Converts to float and scales to [0,1]
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size + 32), # Resize to a bit larger then center crop
        transforms.CenterCrop(img_size),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load the full dataset (without transforms initially to allow splitting first)
    # We will apply transforms per subset.
    full_dataset_no_transform = datasets.ImageFolder(dataset_path)
    class_names = full_dataset_no_transform.classes
    
    dataset_size = len(full_dataset_no_transform)
    indices = list(range(dataset_size))
    np.random.shuffle(indices) # Shuffle indices before splitting

    train_end_idx = int(train_split_ratio * dataset_size)
    val_end_idx = train_end_idx + int(val_split_ratio * dataset_size)

    train_indices = indices[:train_end_idx]
    val_indices = indices[train_end_idx:val_end_idx]
    test_indices = indices[val_end_idx:]
    
    if not test_indices: # Ensure test set is not empty
        if val_indices: # If val exists, take some from val for test
            num_test_from_val = max(1, int(0.1 * len(val_indices))) # e.g. 10% of val or at least 1
            test_indices = val_indices[-num_test_from_val:]
            val_indices = val_indices[:-num_test_from_val]
        elif train_indices: # If only train exists, take some from train for test (and val)
             # This case implies train_split_ratio + val_split_ratio is too high or 1.0
            print("Warning: train_split_ratio + val_split_ratio is too high, deriving test set from train set.")
            num_test_from_train = max(1, int(0.05 * len(train_indices))) # 5% of train for test
            num_val_from_train = max(1, int(0.05 * len(train_indices)))  # 5% of train for val
            test_indices = train_indices[-num_test_from_train:]
            val_indices = train_indices[-(num_test_from_train + num_val_from_train):-num_test_from_train]
            train_indices = train_indices[:-(num_test_from_train + num_val_from_train)]
        else: # Dataset too small
            raise ValueError("Dataset too small or split ratios incorrect, resulting in empty sets.")


    # Create dataset subsets with specific transformations
    # Need to wrap ImageFolder with the transform for each subset
    # This is a bit tricky; an easier way is to create three ImageFolder instances if paths allow,
    # or apply transform inside a custom Subset wrapper or DataLoader's collate_fn.
    # A common way: create a base dataset, then use Subset with a wrapper that applies transform.
    
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_subset_no_transform = Subset(full_dataset_no_transform, train_indices)
    val_subset_no_transform = Subset(full_dataset_no_transform, val_indices)
    test_subset_no_transform = Subset(full_dataset_no_transform, test_indices)

    train_dataset = TransformedSubset(train_subset_no_transform, train_transform)
    val_dataset = TransformedSubset(val_subset_no_transform, eval_transform)
    test_dataset = TransformedSubset(test_subset_no_transform, eval_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"DataLoaders created: Train ({len(train_dataset)} samples), Val ({len(val_dataset)} samples), Test ({len(test_dataset)} samples)")
    return train_loader, val_loader, test_loader, class_names
```

```python
## class TransferLearningLightningModel(pl.LightningModule)
# PyTorch Lightning module for image classification using transfer learning.
# Allows using pre-trained models from torchvision.models (e.g., VGG, ResNet, EfficientNet, ViT).
# Freezes backbone layers and replaces the final classifier head.
# Implements training, validation, test steps, and logs loss & accuracy.
# Includes confusion matrix calculation and plotting at the end of the test phase.
# Used in TP5, TP6, TP7.

# Entrées (__init__) :
# - model_name (str): Name of the torchvision model to use (e.g., "vgg16", "resnet50", "efficientnet_b0", "vit_b_16").
# - num_classes (int): Number of output classes for the new task.
# - class_names (list of str, optional): List of class names for confusion matrix labels. Required if plot_confusion_matrix is True.
# - learning_rate (float, optional): Learning rate. Defaults to 3e-4.
# - optimizer_class (torch.optim.Optimizer, optional): Optimizer class. Defaults to optim.Adam.
# - pretrained_weights (str or torchvision.models.WeightsEnum, optional): Pretrained weights to use.
#   "DEFAULT" uses the best available. None for random initialization. Specific Weight enums can also be passed. Defaults to "DEFAULT".
# - unfreeze_backbone_epoch (int, optional): Epoch at which to unfreeze backbone layers. -1 to keep frozen. Defaults to -1.
# - plot_confusion_matrix_on_test (bool, optional): Whether to plot confusion matrix after test. Defaults to True.


# Sorties :
# - (Implicitly) Manages training, logs metrics.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # For TP7 metrics from confusion matrix

class TransferLearningLightningModel(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, class_names: list = None,
                 learning_rate: float = 3e-4, optimizer_class=optim.Adam,
                 pretrained_weights="DEFAULT", unfreeze_backbone_epoch: int = -1,
                 plot_confusion_matrix_on_test: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]
        self.lr = learning_rate
        self.optimizer_class = optimizer_class
        self.pretrained_weights = pretrained_weights
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self.plot_confusion_matrix_on_test = plot_confusion_matrix_on_test
        
        self.save_hyperparameters() # Saves all __init__ args

        # Load pre-trained model
        if hasattr(models, model_name):
            model_func = getattr(models, model_name)
            self.backbone = model_func(weights=self.pretrained_weights)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")

        # Freeze backbone parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier head
        if "vgg" in model_name:
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif "resnet" in model_name or "resnext" in model_name or "wide_resnet" in model_name or "shufflenet" in model_name or "mobilenet" in model_name:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif "efficientnet" in model_name:
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif "vit" in model_name or "swin" in model_name: # Vision Transformer, Swin Transformer
            if hasattr(self.backbone, 'heads') and isinstance(self.backbone.heads, nn.Sequential) and hasattr(self.backbone.heads, 'head'): # PyTorch 1.12+ ViT
                 in_features = self.backbone.heads.head.in_features
                 self.backbone.heads.head = nn.Linear(in_features, num_classes)
            elif hasattr(self.backbone, 'head') and isinstance(self.backbone.head, nn.Linear): # Older ViT or other transformers
                 in_features = self.backbone.head.in_features
                 self.backbone.head = nn.Linear(in_features, num_classes)
            else: # Fallback for models like ConvNeXt or if structure is unknown
                # This is a guess; manual inspection might be needed for new models
                # Try to find the last linear layer. Common names: 'classifier', 'fc', 'head'
                potential_heads = ['classifier', 'fc', 'head']
                found_head = False
                for head_name in potential_heads:
                    if hasattr(self.backbone, head_name):
                        head_module = getattr(self.backbone, head_name)
                        if isinstance(head_module, nn.Linear):
                            in_features = head_module.in_features
                            setattr(self.backbone, head_name, nn.Linear(in_features, num_classes))
                            found_head = True
                            break
                        elif isinstance(head_module, nn.Sequential) and isinstance(head_module[-1], nn.Linear):
                            in_features = head_module[-1].in_features
                            head_module[-1] = nn.Linear(in_features, num_classes)
                            found_head = True
                            break
                if not found_head:
                    raise ValueError(f"Cannot automatically replace classifier for {model_name}. Please adapt.")

        else:
            raise ValueError(f"Classifier replacement for {model_name} not implemented. Please adapt.")

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        if self.plot_confusion_matrix_on_test:
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        
        self.confmat_for_metrics = None # To store confusion matrix for TP7 Q9

    def forward(self, x):
        return self.backbone(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc_metric = getattr(self, f"{stage}_accuracy")
        acc = acc_metric(logits, y)
        
        self.log(f"{stage}_loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=(stage=="train"), on_epoch=True, prog_bar=True)
        
        if stage == "test" and self.plot_confusion_matrix_on_test:
            self.test_confusion_matrix.update(logits, y)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_start(self):
        if self.unfreeze_backbone_epoch != -1 and self.current_epoch == self.unfreeze_backbone_epoch:
            print(f"Unfreezing backbone at epoch {self.current_epoch}")
            for param in self.backbone.parameters():
                param.requires_grad = True
            # Reconfigure optimizer or adjust learning rates for backbone
            # For simplicity, often the same optimizer continues but with more params to train
            # Or, re-initialize optimizer: self.trainer.optimizers = [self.configure_optimizers()['optimizer']]
            # A common strategy is to use a lower LR for the backbone
            
    def on_test_epoch_end(self):
        if self.plot_confusion_matrix_on_test:
            cm = self.test_confusion_matrix.compute().cpu().numpy()
            self.confmat_for_metrics = cm # Store for TP7 Q9

            fig, ax = plt.subplots(figsize=(max(6, self.num_classes), max(5, self.num_classes*0.8)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            
            # Log to wandb if logger is available
            if self.logger and hasattr(self.logger.experiment, 'log'):
                 self.logger.experiment.log({"test_confusion_matrix": wandb.Image(fig)})
            plt.show()
            self.test_confusion_matrix.reset() # Important to reset for next test run

    def configure_optimizers(self):
        # If unfreezing, you might want different LRs for backbone and head
        # For now, simple optimizer for all trainable parameters
        optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

    def calculate_classification_metrics_from_cm(self):
        """Calculates precision, recall, F1 per class from self.confmat_for_metrics. (For TP7 Q9)"""
        if self.confmat_for_metrics is None:
            print("Confusion matrix not available. Run test phase first.")
            return None

        cm = self.confmat_for_metrics
        num_classes = cm.shape[0]
        metrics = {}

        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            # tn = np.sum(cm) - tp - fp - fn # Not usually needed for per-class P/R/F1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f"Class_{i}"
            metrics[class_name] = {"precision": precision, "recall": recall, "f1_score": f1}
        
        # Macro averages
        avg_precision = np.mean([m['precision'] for m in metrics.values()])
        avg_recall = np.mean([m['recall'] for m in metrics.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics.values()])
        metrics['macro_average'] = {"precision": avg_precision, "recall": avg_recall, "f1_score": avg_f1}
        
        # Micro average (accuracy)
        total_tp = np.sum(np.diag(cm))
        total_samples = np.sum(cm)
        micro_f1_accuracy = total_tp / total_samples if total_samples > 0 else 0
        metrics['micro_average_accuracy'] = micro_f1_accuracy

        df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
        print("\nClassification Metrics per Class:")
        print(df_metrics)
        return df_metrics
```

```python
## def display_class_images_from_path(dataset_path: str, num_images_to_show: int = 5, img_size_display: tuple = (128, 128))
# Displays a few sample images from each class in a dataset structured in subdirectories.
# Used in TP5, TP6.

# Entrées :
# - dataset_path (str): Path to the root directory of the dataset (e.g., "dataset/railway-construction-50/").
# - num_images_to_show (int, optional): Maximum number of images to display per class. Defaults to 5.
# - img_size_display (tuple, optional): Size (width, height) to resize images for display. Defaults to (128, 128).

# Sorties :
# - None (displays the plots).
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # or from PIL import Image; img = Image.open(...)
import cv2 # For resizing if mpimg is used and original TP code uses cv2.resize

def display_class_images_from_path(dataset_path: str, num_images_to_show: int = 5, img_size_display: tuple = (128, 128)):
    """
    Displays a few sample images from each class subfolder in the dataset_path.
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found or not a directory.")
        return

    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not class_names:
        print(f"No subdirectories (classes) found in '{dataset_path}'.")
        return

    for class_name in class_names:
        class_folder_path = os.path.join(dataset_path, class_name)
        # Look for common image extensions
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            image_paths.extend(glob.glob(os.path.join(class_folder_path, ext)))
        
        if not image_paths:
            print(f"No images found in class folder: {class_folder_path}")
            continue

        print(f"\nDisplaying images for class: {class_name}")
        
        # Determine number of columns for subplot dynamically or fix it
        num_cols = min(num_images_to_show, 5) # Max 5 columns
        num_rows = (min(len(image_paths), num_images_to_show) + num_cols - 1) // num_cols


        plt.figure(figsize=(num_cols * 3, num_rows * 3)) # Adjust figsize based on cols/rows
        
        for i, img_path in enumerate(image_paths):
            if i >= num_images_to_show:
                break
            
            try:
                img = mpimg.imread(img_path) # Reads as numpy array
                # Resize for display consistency (OpenCV is good for this)
                img_resized = cv2.resize(img, img_size_display)
                
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(img_resized)
                plt.title(os.path.basename(img_path)[:15]) # Shortened filename
                plt.axis('off')
            except Exception as e:
                print(f"Could not load/display image {img_path}: {e}")
        
        plt.tight_layout()
        plt.show()
```

```python
## def plot_training_metrics_from_csv(log_dir: str, csv_logger_name: str, csv_logger_version: str = '', metrics_filename: str = 'metrics.csv')
# Plots training and validation loss/accuracy from a metrics.csv file generated by PyTorch Lightning's CSVLogger.
# Used in TP5, TP6.

# Entrées :
# - log_dir (str): The base directory where logs are stored (e.g., "logs/").
# - csv_logger_name (str): The 'name' given to CSVLogger (e.g., "cnn1").
# - csv_logger_version (str, optional): The 'version' given to CSVLogger. Defaults to ''.
#                                     If CSVLogger uses subdirectories like 'version_0', 'version_1', include it here.
# - metrics_filename (str, optional): Name of the metrics CSV file. Defaults to 'metrics.csv'.

# Sorties :
# - None (displays the plots).
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics_from_csv(log_dir: str, csv_logger_name: str, csv_logger_version: str = '', metrics_filename: str = 'metrics.csv'):
    """
    Plots training/validation loss and accuracy from CSVLogger's metrics file.
    Assumes standard column names like 'epoch', 'train_loss_epoch', 'val_loss', 
    'train_acc_epoch', 'val_acc'. Adjust if names differ.
    """
    if csv_logger_version: # If version is specified (e.g. "version_0")
        metrics_path = os.path.join(log_dir, csv_logger_name, csv_logger_version, metrics_filename)
    else: # If version is empty, logger might save directly under csv_logger_name
        metrics_path = os.path.join(log_dir, csv_logger_name, metrics_filename)

    if not os.path.exists(metrics_path):
        # Try another common structure if version is managed by PTL automatically (version_0, version_1, etc.)
        # and user provides empty version string. We'll try to find the latest version.
        potential_versions_path = os.path.join(log_dir, csv_logger_name)
        if os.path.isdir(potential_versions_path):
            versions = sorted([d for d in os.listdir(potential_versions_path) if d.startswith("version_") and os.path.isdir(os.path.join(potential_versions_path,d))])
            if versions:
                 metrics_path = os.path.join(potential_versions_path, versions[-1], metrics_filename) # Use latest version

    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at '{metrics_path}' or any auto-detected version path.")
        print("Please check log_dir, csv_logger_name, and csv_logger_version.")
        return

    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"Error reading CSV file '{metrics_path}': {e}")
        return

    # Identify epoch column, often just 'epoch'
    epoch_col = 'epoch' if 'epoch' in df.columns else None
    if not epoch_col:
        print("Error: 'epoch' column not found in metrics.csv.")
        return
        
    # Loss plot
    train_loss_col = next((col for col in ['train_loss_epoch', 'train_loss'] if col in df.columns), None)
    val_loss_col = next((col for col in ['val_loss_epoch', 'val_loss'] if col in df.columns), None)

    if train_loss_col and val_loss_col:
        train_df_loss = df[df[train_loss_col].notna()][[epoch_col, train_loss_col]].copy()
        val_df_loss = df[df[val_loss_col].notna()][[epoch_col, val_loss_col]].copy()
        # PTL might log val metrics less frequently, so forward fill epoch for val if needed for plotting alignment
        # Or group by epoch and take mean if multiple steps per epoch logged for train
        train_df_loss = train_df_loss.groupby(epoch_col).mean().reset_index()
        val_df_loss = val_df_loss.groupby(epoch_col).mean().reset_index()


        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_df_loss[epoch_col], train_df_loss[train_loss_col], label='Train Loss', marker='o')
        plt.plot(val_df_loss[epoch_col], val_df_loss[val_loss_col], label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
    else:
        print("Warning: Loss columns not found for plotting.")

    # Accuracy plot
    train_acc_col = next((col for col in ['train_acc_epoch', 'train_acc'] if col in df.columns), None)
    val_acc_col = next((col for col in ['val_acc_epoch', 'val_acc'] if col in df.columns), None)

    if train_acc_col and val_acc_col:
        train_df_acc = df[df[train_acc_col].notna()][[epoch_col, train_acc_col]].copy()
        val_df_acc = df[df[val_acc_col].notna()][[epoch_col, val_acc_col]].copy()
        train_df_acc = train_df_acc.groupby(epoch_col).mean().reset_index()
        val_df_acc = val_df_acc.groupby(epoch_col).mean().reset_index()

        plt.subplot(1, 2, 2)
        plt.plot(train_df_acc[epoch_col], train_df_acc[train_acc_col], label='Train Accuracy', marker='o')
        plt.plot(val_df_acc[epoch_col], val_df_acc[val_acc_col], label='Validation Accuracy', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.grid(True)
    else:
        print("Warning: Accuracy columns not found for plotting.")
        if not (train_loss_col and val_loss_col): # If no plots were made at all
            return # exit

    plt.tight_layout()
    plt.show()
```

```python
## def generate_kaggle_submission_csv(model: nn.Module, test_data_folder_path: str, image_size: int, output_csv_filename: str = "submission.csv", device_str: str = "auto")
# Generates a CSV file for Kaggle submission by predicting labels for images in a test folder.
# The model is expected to be a PyTorch model. Transformations are applied to test images.
# Used in TP5, TP6.

# Entrées :
# - model (nn.Module): The trained PyTorch model (can be a LightningModule or a raw nn.Module).
# - test_data_folder_path (str): Path to the folder containing test images. Assumes images are directly in this folder.
# - image_size (int): The size (height and width) to which test images will be resized.
# - output_csv_filename (str, optional): Name of the output CSV file. Defaults to "submission.csv".
# - device_str (str, optional): Device to run predictions on ("cpu", "cuda", or "auto"). Defaults to "auto".


# Sorties :
# - submission_df (pd.DataFrame): DataFrame containing image IDs and predicted labels.
# - Saves the submission_df to `output_csv_filename`.
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms # Using v2
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def generate_kaggle_submission_csv(model: nn.Module, test_data_folder_path: str, image_size: int, output_csv_filename: str = "submission.csv", device_str: str = "auto"):
    """
    Generates a Kaggle submission CSV file by predicting labels for images in a test folder.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Standard transformations for test images (consistent with eval_transform)
    mean = [0.485, 0.456, 0.406] # Typical ImageNet mean
    std = [0.229, 0.224, 0.225]  # Typical ImageNet std
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size + 32), # Resize to a bit larger then center crop, or just Resize(image_size)
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True), # Converts to float and scales to [0,1]
        transforms.Normalize(mean=mean, std=std)
    ])

    predictions = []
    filenames = []

    image_files = [f for f in os.listdir(test_data_folder_path) if os.path.isfile(os.path.join(test_data_folder_path, f)) 
                   and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print(f"No image files found in {test_data_folder_path}")
        return pd.DataFrame({'ID': [], 'Label': []})

    for img_filename in tqdm(image_files, desc="Predicting test images"):
        img_path = os.path.join(test_data_folder_path, img_filename)
        try:
            image = Image.open(img_path).convert('RGB') # Ensure 3 channels
            image_tensor = test_transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

            with torch.no_grad():
                output_logits = model(image_tensor)
                _, predicted_class_idx = torch.max(output_logits, 1) # Get index of max logit

            predictions.append(predicted_class_idx.item())
            filenames.append(img_filename) # Kaggle usually wants the filename as ID
        except Exception as e:
            print(f"Could not process image {img_path}: {e}")
            # Optionally append a placeholder prediction or skip
            # predictions.append(-1) # Placeholder for error
            # filenames.append(img_filename)


    submission_df = pd.DataFrame({
        'ID': filenames,    # Or 'id' depending on Kaggle requirements
        'Label': predictions # Or 'target', 'class', etc.
    })

    submission_df.to_csv(output_csv_filename, index=False)
    print(f"Submission file '{output_csv_filename}' created successfully with {len(submission_df)} predictions.")
    return submission_df
```

```python
## def visualize_vit_patches_and_embeddings(vit_model: nn.Module, image_path: str, image_size: int = 224)
# Visualizes the patch embedding process of a Vision Transformer (ViT) model.
# It shows:
# 1. The original image.
# 2. The image divided into patches.
# 3. The similarity of positional embeddings.
# This function is primarily for educational understanding of ViT's input processing.
# Adapted from TP6.

# Entrées :
# - vit_model (nn.Module): An instance of a ViT model (e.g., from torchvision.models or a custom one
#                          that has `conv_proj`, `encoder.pos_embedding`, `class_token`).
# - image_path (str): Path to the input image file.
# - image_size (int, optional): Size to resize the image to. Defaults to 224.


# Sorties :
# - None (displays plots).
# - Prints shapes of intermediate tensors.
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms # Using v2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 # For image manipulation in plots

def visualize_vit_patches_and_embeddings(vit_model_instance: nn.Module, image_path: str, image_size: int = 224):
    """
    Visualizes patch embedding and positional embedding similarity for a ViT model.
    Assumes vit_model_instance is a PyTorch ViT model (e.g., from torchvision or similar structure).
    """
    if not (hasattr(vit_model_instance, 'conv_proj') and 
            hasattr(vit_model_instance, 'encoder') and 
            hasattr(vit_model_instance.encoder, 'pos_embedding') and
            hasattr(vit_model_instance, 'class_token')):
        print("Warning: The provided model doesn't seem to have the expected ViT attributes "
              "('conv_proj', 'encoder.pos_embedding', 'class_token'). Visualization might fail or be inaccurate.")
        # return # Or try to proceed cautiously

    # --- 1. Load and transform image ---
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # Transformation for ViT input
    transform_to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Resize
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True), # To float tensor [0,1]
        # ViT models in torchvision often have their own normalization or expect [0,1]
        # If using a specific pretrained ViT, check its required normalization
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor_transformed = transform_to_tensor(img_pil).unsqueeze(0) # Add batch dimension
    
    print(f"Original Image PIL size: {img_pil.size}")
    print(f"Transformed Image Tensor shape: {img_tensor_transformed.shape}")

    # --- 2. Patch Embedding ---
    # Ensure model is in eval mode for consistency, though not strictly necessary for this part
    vit_model_instance.eval() 
    with torch.no_grad():
        # `conv_proj` is typical for PyTorch's ViT: Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        if hasattr(vit_model_instance, 'conv_proj'):
            patch_embeddings_conv = vit_model_instance.conv_proj(img_tensor_transformed) # Shape: [1, embed_dim, num_patches_h, num_patches_w]
            print(f"Patch Embeddings (after conv_proj) shape: {patch_embeddings_conv.shape}")
            
            # For visualization, we need patch size. Infer from conv_proj
            patch_size_h = vit_model_instance.conv_proj.kernel_size[0]
            patch_size_w = vit_model_instance.conv_proj.kernel_size[1]
            num_patches_h = image_size // patch_size_h
            num_patches_w = image_size // patch_size_w
            print(f"Inferred patch size: ({patch_size_h}x{patch_size_w}), Num patches: ({num_patches_h}x{num_patches_w})")
        else:
            print("Model does not have 'conv_proj'. Skipping patch embedding visualization details.")
            return


    # --- 3. Visualize Patches on Original Image ---
    img_display_np = np.array(img_pil.resize((image_size, image_size))) # For display

    fig_patches = plt.figure(figsize=(8, 8))
    fig_patches.suptitle("Image Divided into Patches", fontsize=16)
    for i in range(num_patches_h * num_patches_w):
        row = i // num_patches_w
        col = i % num_patches_w
        
        patch = img_display_np[row*patch_size_h:(row+1)*patch_size_h, col*patch_size_w:(col+1)*patch_size_w]
        
        ax = fig_patches.add_subplot(num_patches_h, num_patches_w, i + 1)
        ax.imshow(patch)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()

    # --- 4. Positional Embeddings ---
    if hasattr(vit_model_instance, 'encoder') and hasattr(vit_model_instance.encoder, 'pos_embedding'):
        pos_embed = vit_model_instance.encoder.pos_embedding # Shape: [1, num_patches + 1, embed_dim] (+1 for class token)
        print(f"Positional Embeddings shape: {pos_embed.shape}")

        # Visualize similarity of positional embeddings (excluding class token's embedding)
        # pos_embed_patches = pos_embed[0, 1:, :] # Exclude class token
        # num_patches_total = pos_embed_patches.shape[0]
        
        # This part assumes num_patches_h * num_patches_w == num_patches_total
        # which holds if pos_embed corresponds to the grid
        if num_patches_h * num_patches_w == pos_embed.shape[1] -1 : # Check if grid matches pos_embed
            pos_embed_patches = pos_embed[0, 1:, :] # Exclude class token
            
            fig_pos_sim = plt.figure(figsize=(10, 10))
            fig_pos_sim.suptitle("Positional Embedding Similarities to Others", fontsize=16)
            
            # Plot cosine similarity of each patch's pos_embed to all other patch pos_embeds
            for i in range(num_patches_h * num_patches_w):
                # Cosine similarity of i-th patch pos_embed with all patch pos_embeds
                sim_matrix = F.cosine_similarity(pos_embed_patches[i:i+1, :], pos_embed_patches, dim=1)
                sim_matrix_reshaped = sim_matrix.reshape((num_patches_h, num_patches_w)).detach().cpu().numpy()
                
                ax = fig_pos_sim.add_subplot(num_patches_h, num_patches_w, i + 1)
                ax.imshow(sim_matrix_reshaped, cmap='viridis')
                ax.set_title(f"Patch {i}", fontsize=8)
                ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("Number of patches from image division does not match positional embedding dimensions. Skipping similarity plot.")

    else:
        print("Model does not have 'encoder.pos_embedding'. Skipping positional embedding visualization.")


    # --- 5. Combine Patches with Positional Embeddings (Conceptual from TP) ---
    if hasattr(vit_model_instance, 'conv_proj') and \
       hasattr(vit_model_instance, 'class_token') and \
       hasattr(vit_model_instance.encoder, 'pos_embedding'):
        
        # Patches are [1, embed_dim, num_patches_h, num_patches_w]
        # Flatten and permute to [1, num_patches_h*num_patches_w, embed_dim]
        patches_flattened = patch_embeddings_conv.flatten(2).transpose(1, 2) 
        
        cls_token = vit_model_instance.class_token # Shape: [1, 1, embed_dim]
        
        # Concatenate class token with patch embeddings
        transformer_input_no_pos = torch.cat((cls_token, patches_flattened), dim=1) # Shape [1, num_patches+1, embed_dim]
        print(f"Transformer Input (CLS token + flattened patches) shape: {transformer_input_no_pos.shape}")
        
        # Add positional embeddings
        # pos_embed is [1, num_patches+1, embed_dim]
        if transformer_input_no_pos.shape == pos_embed.shape:
            transformer_input_with_pos = transformer_input_no_pos + pos_embed
            print(f"Transformer Input (with positional embeddings) shape: {transformer_input_with_pos.shape}")
        else:
            print(f"Shape mismatch: Input without pos_embed {transformer_input_no_pos.shape} vs pos_embed {pos_embed.shape}")

# Example (requires a ViT model and an image):
# vit_b16_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
# dummy_image_path = "path_to_your_sample_image.jpg" # Create a dummy image or use one from dataset
# with open(dummy_image_path, 'wb') as f: # Create a dummy image if none exists
#    f.write(os.urandom(1024)) # Minimal content, PIL might complain
# try:
#    Image.new('RGB', (224, 224)).save(dummy_image_path) # A proper blank image
#    visualize_vit_patches_and_embeddings(vit_b16_model, dummy_image_path)
# except Exception as e:
#    print(f"Error in ViT visualization example: {e}")
```

```python
## def benchmark_pytorch_model(model: nn.Module, example_input_shape: tuple = (1, 3, 224, 224), gpu_only: bool = True, device_str: str = "auto")
# Benchmarks a PyTorch model for various performance metrics like inference time, memory usage, FLOPs etc.
# Uses the `pytorch_bench` library (assumed to be installed).
# From TP6.

# Entrées :
# - model (nn.Module): The PyTorch model to benchmark.
# - example_input_shape (tuple, optional): Shape of an example input tensor (Batch, Channels, Height, Width).
#   Defaults to (1, 3, 224, 224).
# - gpu_only (bool, optional): If True, run benchmark only on GPU (if available). Passed to `pytorch_bench.benchmark`.
#   Defaults to True.
# - device_str (str, optional): Device to create example input on ("cpu", "cuda", "auto"). Defaults to "auto".


# Sorties :
# - results (dict or relevant type from pytorch_bench): Benchmark results.
# - Prints the results.
import torch
import torch.nn as nn
# from pytorch_bench import benchmark # Assuming this is installed

def benchmark_pytorch_model(model: nn.Module, example_input_shape: tuple = (1, 3, 224, 224), gpu_only: bool = True, device_str: str = "auto"):
    """
    Benchmarks a PyTorch model using the pytorch_bench library.
    Ensure pytorch_bench is installed: pip install pytorch-bench
    """
    try:
        from pytorch_bench import benchmark as run_benchmark
    except ImportError:
        print("Error: pytorch_bench library not found. Please install it: pip install pytorch-bench")
        print("Skipping benchmark.")
        return None

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if gpu_only and device.type == 'cpu':
        print("GPU not available, but gpu_only=True. Skipping benchmark or running on CPU if pytorch_bench handles it.")
        # pytorch_bench might run on CPU if GPU is not found, even if gpu_only=True.
        # Or, you might want to explicitly skip:
        # return None 

    model.to(device)
    model.eval() # Ensure model is in eval mode

    # Create example input tensor on the correct device
    example_input = torch.randn(example_input_shape).to(device)

    print(f"Benchmarking model {model.__class__.__name__} on {device.type.upper()} with input shape {example_input_shape}...")
    
    try:
        # The `benchmark` function from `pytorch_bench` might have specific arguments.
        # The TP shows `gpu_only=True`. Check library docs for exact signature if issues arise.
        results = run_benchmark(model, example_input, gpu_only=(gpu_only and device.type=='cuda')) # Pass gpu_only based on actual device
        print("\nBenchmark Results:")
        # `results` format depends on pytorch_bench; typically a dict or custom object
        if isinstance(results, dict):
            for key, value in results.items():
                print(f"  {key}: {value}")
        else:
            print(results)
        return results
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return None

# Example (commented out, requires a model):
# my_cnn_model = models.resnet18() 
# benchmark_results = benchmark_pytorch_model(my_cnn_model)
```

```python
## def get_test_images_by_class_from_dataloader(test_loader: DataLoader, num_images_per_class: int = 5) -> dict
# Extracts a specified number of test images (and their labels/paths if available from dataset)
# for each class from a PyTorch DataLoader.
# Useful for qualitative analysis or XAI.
# From TP7.

# Entrées :
# - test_loader (DataLoader): DataLoader for the test set. The underlying dataset should be
#                             an ImageFolder or a Subset of an ImageFolder.
# - num_images_per_class (int, optional): Number of images to extract per class. Defaults to 5.

# Sorties :
# - images_by_class (dict): A dictionary where keys are class names and values are lists of tuples.
#                           Each tuple: (image_tensor, true_label_idx, image_path_if_available)
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets # To check instance type
from collections import defaultdict

def get_test_images_by_class_from_dataloader(test_loader: DataLoader, num_images_per_class: int = 5) -> dict:
    """
    Extracts and organizes test images by class from a DataLoader.
    Attempts to get image paths if the dataset is an ImageFolder.
    """
    # Initialize dictionary to store images by class index
    # Using defaultdict for convenience
    images_by_class_idx = defaultdict(list)
    
    # Determine class names and class_to_idx mapping
    # This requires inspecting the dataset structure within the DataLoader
    if hasattr(test_loader.dataset, 'dataset') and isinstance(test_loader.dataset.dataset, datasets.ImageFolder):
        # Case: DataLoader wraps a Subset of an ImageFolder
        original_dataset = test_loader.dataset.dataset
        class_names = original_dataset.classes
        class_to_idx = original_dataset.class_to_idx
        is_imagefolder_subset = True
        # We need subset indices to map back to original dataset samples for paths
        subset_indices = test_loader.dataset.indices
        
        # Create a map from subset_idx to original_dataset_idx
        # This is tricky if DataLoader shuffles. Assuming it doesn't for test_loader.
        # If test_loader.sampler is RandomSampler or shuffle=True, this direct mapping is not reliable for paths.
        # For simplicity, if shuffle is on, paths might not be correctly associated.
        # TP7 code iterates through subset_indices directly. Let's try that.
        
    elif isinstance(test_loader.dataset, datasets.ImageFolder):
        # Case: DataLoader wraps an ImageFolder directly
        original_dataset = test_loader.dataset
        class_names = original_dataset.classes
        class_to_idx = original_dataset.class_to_idx
        is_imagefolder_subset = False
        subset_indices = list(range(len(original_dataset))) # All indices
    else:
        print("Warning: Dataset type not ImageFolder or Subset of ImageFolder. Cannot retrieve class names or paths automatically.")
        # Fallback: use label indices as keys if class names are unknown
        # We won't be able to get paths in this case.
        class_names = None 
        class_to_idx = None
        is_imagefolder_subset = False # or unknown

    # Iterate through the DataLoader
    # Note: Iterating through DataLoader is easier than managing indices if paths are not strictly needed
    # or if the transform is complex.
    # However, TP7's `explain_model` needs the path.
    
    collected_counts = defaultdict(int)
    
    # Iterate through the DataLoader once to collect images
    # This assumes test_loader.shuffle is False (typical for test sets)
    temp_storage = defaultdict(list)
    for batch_images, batch_labels in test_loader:
        for i in range(len(batch_images)):
            img_tensor = batch_images[i]
            label_idx = batch_labels[i].item()
            
            if collected_counts[label_idx] < num_images_per_class:
                 # Path retrieval is complex here without original index from dataset.
                 # For now, store (img_tensor, label_idx, None_for_path)
                temp_storage[label_idx].append((img_tensor, label_idx, None)) # Path placeholder
                collected_counts[label_idx] +=1
        
        # Check if we have enough images for all classes we've seen so far
        # This logic might not get N images for *every* class if some classes are rare
        # A better approach is to iterate over the dataset directly using indices if paths are needed.
        # Let's refine based on TP7's `get_test_images_by_class` which iterates `subset_indices`.
        
    # Re-doing collection based on TP7's direct dataset iteration for path access
    images_by_class_idx.clear() # Clear previous attempt
    
    if is_imagefolder_subset or (not is_imagefolder_subset and isinstance(original_dataset, datasets.ImageFolder)):
        # Iterate through the indices of the test set
        current_idx_in_subset = 0
        for original_idx in subset_indices: # These are indices into `original_dataset`
            if current_idx_in_subset >= test_loader.batch_size * len(test_loader) : # Avoid iterating too much if subset is smaller than full iteration
                break

            img_tensor, label_idx = original_dataset[original_idx] # This applies transform
            path = original_dataset.samples[original_idx][0] if hasattr(original_dataset, 'samples') else None
            
            if len(images_by_class_idx[label_idx]) < num_images_per_class:
                images_by_class_idx[label_idx].append((img_tensor, label_idx, path))
            
            current_idx_in_subset += 1
            # Check if all classes have enough images
            all_filled = True
            # Check against actual classes present in the original_dataset
            for known_label_idx in range(len(class_names if class_names else [])): 
                if len(images_by_class_idx[known_label_idx]) < num_images_per_class:
                    all_filled = False
                    break
            if all_filled and class_names: # Ensure class_names is not None
                break 
    else: # Fallback to temp_storage if not ImageFolder (paths will be None)
        images_by_class_idx = temp_storage


    # Convert to dict with class names as keys if possible
    final_images_by_class = {}
    if class_names:
        for label_idx, items_list in images_by_class_idx.items():
            final_images_by_class[class_names[label_idx]] = items_list
    else: # Use label index as key if class names are not known
        final_images_by_class = dict(images_by_class_idx)
        
    print("\nExtracted test images by class:")
    for c_name, images_list in final_images_by_class.items():
        print(f"  Class '{c_name}': {len(images_list)} images")
        # for img_tensor, label, path_str in images_list:
        #     print(f"    Label: {label}, Path: {path_str if path_str else 'N/A'}")
            
    return final_images_by_class
```

```python
## def explain_cnn_model_with_captum_torchcam(cnn_model: nn.Module, image_tensor_batch: torch.Tensor, true_label_idx: int, class_names: list, xai_method_name: str, image_path: str = None, device_str: str = "auto")
# Explains a CNN model's prediction on a single image using specified XAI methods
# from Captum (GradientSHAP, Occlusion, IntegratedGradients, DeepLift) or
# TorchCAM (GradCAM, LayerCAM).
# From TP7.

# Entrées :
# - cnn_model (nn.Module): The trained PyTorch CNN model (e.g., an EfficientNet instance).
#                         Assumes it's a `TransferLearningLightningModel` or has `model.features[-1]` for CAM methods.
# - image_tensor_batch (torch.Tensor): Input image tensor, expected shape (1, C, H, W).
# - true_label_idx (int): The true class index of the input image.
# - class_names (list): List of class names, where index corresponds to class index.
# - xai_method_name (str): Name of the XAI method to use.
#   Supported: "gradientshap", "occlusion", "integratedgradients", "deeplift", "gradcam", "layercam".
# - image_path (str, optional): Path to the original image file (used for one of the subplots). Defaults to None.
# - device_str (str, optional): Device to run on ("cpu", "cuda", "auto"). Defaults to "auto".

# Sorties :
# - None (displays a plot with original image, XAI attribution map, overlay, and prediction).
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image # For overlay_mask
from PIL import Image # For overlay_mask
import cv2 # For text on image

# XAI libraries
from torchcam.methods import GradCAM, LayerCAM
from torchcam.utils import overlay_mask # TorchCAM's overlay
from captum.attr import GradientShap, Occlusion, IntegratedGradients, DeepLift 
# captum.attr.visualization can also be used, but TP7 uses custom plotting

def explain_cnn_model_with_captum_torchcam(
    cnn_model: nn.Module, 
    image_tensor_batch: torch.Tensor, # Expects a batch of 1 image: (1, C, H, W)
    true_label_idx: int, 
    class_names: list, 
    xai_method_name: str, 
    image_path: str = None, # Path to original image for one of the plots
    device_str: str = "auto"):
    """
    Explains a CNN model's prediction using Captum or TorchCAM methods.
    `cnn_model` can be a raw nn.Module or a pl.LightningModule.
    `image_tensor_batch` should be a single image tensor with batch dim: (1, C, H, W).
    """

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Ensure model is on the correct device and in eval mode
    # If cnn_model is a LightningModule, the actual nn.Module is often at self.model or self.backbone
    # For simplicity, assume cnn_model itself is the nn.Module or can be called directly.
    # If it's a LightningModule, you might need to pass cnn_model.model or cnn_model.backbone
    # to CAM extractors if they need specific layer names from the *inner* model.
    
    # Check if cnn_model is a LightningModule and get the underlying nn.Module if so for CAMs.
    # This is a common pattern if the LightningModule wraps the actual network.
    inner_model_for_cam = cnn_model
    if isinstance(cnn_model, pl.LightningModule):
        if hasattr(cnn_model, 'model') and isinstance(cnn_model.model, nn.Module): # Common for custom LightningModules
            inner_model_for_cam = cnn_model.model
        elif hasattr(cnn_model, 'backbone') and isinstance(cnn_model.backbone, nn.Module): # Common for TransferLearningLM
             inner_model_for_cam = cnn_model.backbone
    
    cnn_model.to(device)
    cnn_model.eval()
    
    # Ensure image tensor is on the correct device and requires grad for gradient-based methods
    img_input = image_tensor_batch.clone().detach().to(device).requires_grad_(True)

    # Get model prediction
    with torch.no_grad(): # No gradients needed for this initial prediction
        output_logits = cnn_model(img_input) # Shape: (1, num_classes)
    
    output_probs = F.softmax(output_logits, dim=1)
    prediction_score, pred_label_idx_tensor = torch.topk(output_probs, 1)
    pred_label_idx = pred_label_idx_tensor.item() # Scalar index

    attributions_np = None # To store the final attribution map as numpy array

    # --- Captum Methods ---
    if xai_method_name.lower() == "gradientshap":
        gradient_shap = GradientShap(cnn_model) # Pass the callable model
        # Baseline: random noise, or zeros, or blurred image, etc.
        rand_img_dist = torch.cat([img_input * 0, img_input * 1]) # Example: zeros and ones as baselines
        # rand_img_dist = torch.randn_like(img_input).repeat(20,1,1,1).to(device) # For more baselines
        attr_tensor = gradient_shap.attribute(img_input,
                                              baselines=rand_img_dist, #expects baselines to be a tensor or tuple of tensors
                                              target=pred_label_idx,
                                              n_samples=10, stdevs=0.001) # n_samples for SmoothGrad-like behavior
        attributions_np = attr_tensor.sum(dim=1).squeeze(0).cpu().detach().numpy() # Sum over channels, remove batch

    elif xai_method_name.lower() == "occlusion":
        occlusion = Occlusion(cnn_model)
        attr_tensor = occlusion.attribute(img_input,
                                          target=pred_label_idx,
                                          strides=(3, 8, 8), # (channels, H, W)
                                          sliding_window_shapes=(3, 15, 15), # Must match channels
                                          baselines=0) # Occlude with zeros
        attributions_np = attr_tensor.sum(dim=1).squeeze(0).cpu().detach().numpy()

    elif xai_method_name.lower() == "integratedgradients":
        ig = IntegratedGradients(cnn_model)
        baseline = torch.zeros_like(img_input).to(device)
        attr_tensor = ig.attribute(img_input, baselines=baseline, target=pred_label_idx)
        attributions_np = attr_tensor.sum(dim=1).squeeze(0).cpu().detach().numpy()

    elif xai_method_name.lower() == "deeplift":
        dl = DeepLift(cnn_model)
        baseline = torch.zeros_like(img_input).to(device)
        attr_tensor = dl.attribute(img_input, baselines=baseline, target=pred_label_idx)
        attributions_np = attr_tensor.sum(dim=1).squeeze(0).cpu().detach().numpy()

    # --- TorchCAM Methods ---
    elif xai_method_name.lower() in ["gradcam", "layercam"]:
        # CAM methods need a target layer from the *inner_model_for_cam*
        # Assuming a common structure for torchvision models (e.g., .features for VGG/EfficientNet, .layer4 for ResNet)
        target_layer_name = None
        if hasattr(inner_model_for_cam, 'features') and isinstance(inner_model_for_cam.features, nn.Sequential): # VGG, EfficientNet
            target_layer_name = inner_model_for_cam.features[-1] # Often the last block of features
            if isinstance(target_layer_name, nn.AdaptiveAvgPool2d): # For EfficientNet, pool is not a conv/bn block
                 target_layer_name = inner_model_for_cam.features[-2] 
        elif hasattr(inner_model_for_cam, 'layer4') and isinstance(inner_model_for_cam.layer4, nn.Sequential): # ResNet
            target_layer_name = inner_model_for_cam.layer4[-1]
        else:
            print(f"Warning: Could not automatically determine target layer for CAM method on {inner_model_for_cam.__class__.__name__}. Using model directly.")
            # This might cause issues if TorchCAM expects a specific layer module.

        if not target_layer_name:
            print("Error: Target layer for CAM could not be found. CAM methods will fail.")
            return

        if xai_method_name.lower() == "gradcam":
            cam_extractor = GradCAM(inner_model_for_cam, target_layer=target_layer_name)
        else: # layercam
            cam_extractor = LayerCAM(inner_model_for_cam, target_layer=target_layer_name)
        
        # CAM extractors usually work with the raw model (not the LightningModule)
        # And they need the class index of the prediction.
        # The output_logits here are from the cnn_model (which might be LightningModule)
        # If cnn_model is Lightning, its output is already logits.
        # If inner_model_for_cam is different, we might need its specific output.
        # For simplicity, use output_logits from the main model call.
        
        activation_map_list = cam_extractor(output_logits.squeeze(0).argmax().item(), output_logits) # Pass class_idx and scores
        cam_extractor.remove_hooks() # Important!
        if activation_map_list and len(activation_map_list) > 0:
            attributions_np = activation_map_list[0].cpu().numpy() # Already [H, W]
        else:
            print("Error: CAM extractor did not return an activation map.")
            return
    else:
        raise ValueError(f"Unsupported XAI method: {xai_method_name}")

    # --- Plotting ---
    if attributions_np is None:
        print("Attribution map not generated.")
        return
        
    # Normalize attributions for display
    attributions_norm = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min() + 1e-9) # Add epsilon

    original_img_pil = to_pil_image(img_input.squeeze(0).cpu()) # For overlay_mask
    attribution_map_pil = to_pil_image(attributions_norm, mode='F') # Mode 'F' for float grayscale

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_img_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(attributions_norm, cmap='jet')
    axes[1].set_title(f"{xai_method_name.capitalize()} Map")
    axes[1].axis('off')

    axes[2].imshow(overlay_mask(original_img_pil, attribution_map_pil, alpha=0.5))
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Prediction text on original image (if path provided)
    actual_class_name = class_names[true_label_idx]
    predicted_class_name = class_names[pred_label_idx]
    is_correct = (predicted_class_name == actual_class_name)

    if image_path and os.path.exists(image_path):
        img_for_text_cv2 = cv2.imread(image_path)
        img_for_text_cv2 = cv2.cvtColor(img_for_text_cv2, cv2.COLOR_BGR2RGB)
    else: # Fallback to tensor if path not good
        img_for_text_cv2 = np.array(original_img_pil) 

    # Resize for consistent text size if needed, or adjust font scale dynamically
    # img_for_text_cv2 = cv2.resize(img_for_text_cv2, (224,224)) # Example fixed size

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(img_for_text_cv2.shape[0], img_for_text_cv2.shape[1]) / 400 # Dynamic font scale
    thickness = max(1, int(font_scale * 2))
    
    pred_text = f"Pred: {predicted_class_name} ({prediction_score.item():.2f})"
    true_text = f"True: {actual_class_name}"

    # Get text size to position it
    (tw1, th1), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(true_text, font, font_scale, thickness)

    cv2.putText(img_for_text_cv2, pred_text, (10, th1 + 10), font, font_scale, (0, 255, 0) if is_correct else (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img_for_text_cv2, true_text, (10, th1 + th2 + 20), font, font_scale, (255,255,255), thickness, cv2.LINE_AA) # White for true label

    axes[3].imshow(img_for_text_cv2)
    axes[3].set_title("Prediction")
    axes[3].axis('off')

    fig.suptitle(f"XAI: {xai_method_name.capitalize()} | Correct: {is_correct}", fontsize=16, color='green' if is_correct else 'red')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle
    plt.show()
```

```python
## def explain_vit_model_with_captum_tis(vit_model: nn.Module, image_tensor_batch: torch.Tensor, true_label_idx: int, class_names: list, xai_method_name: str, image_path: str = None, device_str: str = "auto", tis_n_masks: int = 1024, tis_batch_size: int = 128)
# Explains a Vision Transformer (ViT) model's prediction on a single image.
# Uses specified XAI methods from Captum (GradientSHAP, Occlusion, IntegratedGradients, DeepLift)
# or the TIS (Transformer Input Sampling) method.
# From TP7.

# Entrées :
# - vit_model (nn.Module): The trained PyTorch ViT model.
#                         Assumes it's a `TransferLearningLightningModel` or a raw ViT.
# - image_tensor_batch (torch.Tensor): Input image tensor, shape (1, C, H, W).
# - true_label_idx (int): The true class index of the input image.
# - class_names (list): List of class names.
# - xai_method_name (str): Name of the XAI method.
#   Supported: "gradientshap", "occlusion", "integratedgradients", "deeplift", "tis".
# - image_path (str, optional): Path to the original image file for plotting. Defaults to None.
# - device_str (str, optional): Device ("cpu", "cuda", "auto"). Defaults to "auto".
# - tis_n_masks (int, optional): Number of masks for TIS. Defaults to 1024.
# - tis_batch_size (int, optional): Batch size for TIS. Defaults to 128.

# Sorties :
# - None (displays a plot with original image, XAI attribution, overlay, and prediction).
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import cv2
import os
import numpy as np

# XAI libraries
from captum.attr import GradientShap, Occlusion, IntegratedGradients, DeepLift
# TIS requires cloning/installing its specific library (done in TP7 instructions)
# from Transformer_Input_Sampling.tis import TIS # Assuming it's importable

def explain_vit_model_with_captum_tis(
    vit_model: nn.Module, 
    image_tensor_batch: torch.Tensor, # Expects a batch of 1 image: (1, C, H, W)
    true_label_idx: int, 
    class_names: list, 
    xai_method_name: str, 
    image_path: str = None, 
    device_str: str = "auto",
    tis_n_masks: int = 1024, 
    tis_batch_size: int = 128):
    """
    Explains a ViT model's prediction using Captum or TIS methods.
    `vit_model` can be a raw nn.Module or a pl.LightningModule.
    `image_tensor_batch` should be a single image tensor with batch dim: (1, C, H, W).
    """
    
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    vit_model.to(device)
    vit_model.eval()
    
    img_input = image_tensor_batch.clone().detach().to(device).requires_grad_(True)

    with torch.no_grad():
        output_logits = vit_model(img_input)
    output_probs = F.softmax(output_logits, dim=1)
    prediction_score, pred_label_idx_tensor = torch.topk(output_probs, 1)
    pred_label_idx = pred_label_idx_tensor.item()

    attributions_np = None
    attr_tensor = None # Define for scope

    # --- Captum Methods (same as for CNN) ---
    if xai_method_name.lower() == "gradientshap":
        gradient_shap = GradientShap(vit_model)
        rand_img_dist = torch.cat([img_input * 0, img_input * 1]) 
        attr_tensor = gradient_shap.attribute(img_input, baselines=rand_img_dist, target=pred_label_idx, n_samples=10, stdevs=0.001)
    elif xai_method_name.lower() == "occlusion":
        occlusion = Occlusion(vit_model)
        attr_tensor = occlusion.attribute(img_input, target=pred_label_idx, strides=(3, 16, 16), # Patch size for ViT typically 16
                                          sliding_window_shapes=(3, 16, 16), baselines=0) # Occlude one patch at a time
    elif xai_method_name.lower() == "integratedgradients":
        ig = IntegratedGradients(vit_model)
        baseline = torch.zeros_like(img_input).to(device)
        attr_tensor = ig.attribute(img_input, baselines=baseline, target=pred_label_idx)
    elif xai_method_name.lower() == "deeplift":
        dl = DeepLift(vit_model)
        baseline = torch.zeros_like(img_input).to(device)
        attr_tensor = dl.attribute(img_input, baselines=baseline, target=pred_label_idx)
    
    if attr_tensor is not None: # If a Captum method was used
         attributions_np = attr_tensor.sum(dim=1).squeeze(0).cpu().detach().numpy() # Sum over channels

    # --- TIS Method ---
    elif xai_method_name.lower() == "tis":
        try:
            from Transformer_Input_Sampling.tis import TIS # Local import
            
            # TIS expects the raw nn.Module, not the LightningModule wrapper
            inner_vit_model = vit_model
            if isinstance(vit_model, pl.LightningModule):
                if hasattr(vit_model, 'model') and isinstance(vit_model.model, nn.Module):
                    inner_vit_model = vit_model.model
                elif hasattr(vit_model, 'backbone') and isinstance(vit_model.backbone, nn.Module): # Common for TransferLearningLM
                     inner_vit_model = vit_model.backbone
                else:
                    print("Warning: Could not get inner nn.Module from LightningModule for TIS. Using the LightningModule itself.")
            
            tis_explainer = TIS(inner_vit_model, n_masks=tis_n_masks, batch_size=tis_batch_size, verbose=False) # verbose=False to reduce print
            # TIS returns attribution map directly, shape (H, W) or (1, H, W)
            attr_map_tis = tis_explainer(img_input).cpu().detach() # img_input already on device
            if attr_map_tis.ndim == 3 and attr_map_tis.shape[0] == 1: # If [1, H, W]
                attributions_np = attr_map_tis.squeeze(0).numpy()
            elif attr_map_tis.ndim == 2: # If [H, W]
                 attributions_np = attr_map_tis.numpy()
            else:
                print(f"Unexpected TIS output shape: {attr_map_tis.shape}")
                return
        except ImportError:
            print("Error: Transformer_Input_Sampling (TIS) library not found or import failed.")
            print("Please ensure it's installed and accessible (e.g., cloned from GitHub as per TP7).")
            return
        except Exception as e:
            print(f"Error during TIS explanation: {e}")
            return
    else:
        raise ValueError(f"Unsupported XAI method for ViT: {xai_method_name}")

    # --- Plotting (same as for CNN) ---
    if attributions_np is None:
        print("Attribution map not generated.")
        return
        
    attributions_norm = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min() + 1e-9)

    original_img_pil = to_pil_image(img_input.squeeze(0).cpu())
    attribution_map_pil = to_pil_image(attributions_norm, mode='F')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_img_pil); axes[0].set_title("Original Image"); axes[0].axis('off')
    axes[1].imshow(attributions_norm, cmap='jet'); axes[1].set_title(f"{xai_method_name.capitalize()} Map"); axes[1].axis('off')
    
    # For overlay_mask, need PIL images. TorchCAM's overlay_mask is convenient.
    # If not using torchcam's overlay, implement a simple one:
    # overlayed_img = np.array(original_img_pil) * 0.5 + plt.cm.jet(attributions_norm)[:,:,:3] * 255 * 0.5
    # axes[2].imshow(overlayed_img.astype(np.uint8))
    try:
        from torchcam.utils import overlay_mask as torchcam_overlay # Try to use torchcam's
        axes[2].imshow(torchcam_overlay(original_img_pil, attribution_map_pil, alpha=0.5))
    except ImportError: # Fallback if torchcam not available or issues with its overlay
        print("TorchCAM overlay not available, showing simple heatmap blend.")
        blend = np.array(original_img_pil, dtype=float) * 0.5
        heatmap_color = (plt.cm.jet(attributions_norm)[:, :, :3] * 255).astype(float) # Get RGB from cmap
        blend += heatmap_color * 0.5
        axes[2].imshow(np.clip(blend, 0, 255).astype(np.uint8))

    axes[2].set_title("Overlay"); axes[2].axis('off')
    
    actual_class_name = class_names[true_label_idx]
    predicted_class_name = class_names[pred_label_idx]
    is_correct = (predicted_class_name == actual_class_name)

    if image_path and os.path.exists(image_path):
        img_for_text_cv2 = cv2.imread(image_path)
        img_for_text_cv2 = cv2.cvtColor(img_for_text_cv2, cv2.COLOR_BGR2RGB)
    else:
        img_for_text_cv2 = np.array(original_img_pil)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(img_for_text_cv2.shape[0], img_for_text_cv2.shape[1]) / 400
    thickness = max(1, int(font_scale * 2))
    pred_text = f"Pred: {predicted_class_name} ({prediction_score.item():.2f})"
    true_text = f"True: {actual_class_name}"
    (tw1, th1), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(true_text, font, font_scale, thickness)
    cv2.putText(img_for_text_cv2, pred_text, (10, th1 + 10), font, font_scale, (0, 255, 0) if is_correct else (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img_for_text_cv2, true_text, (10, th1 + th2 + 20), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    axes[3].imshow(img_for_text_cv2); axes[3].set_title("Prediction"); axes[3].axis('off')
    fig.suptitle(f"XAI (ViT): {xai_method_name.capitalize()} | Correct: {is_correct}", fontsize=16, color='green' if is_correct else 'red')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
```

```python
## def create_rise_masks(num_masks: int, mask_small_h: int, mask_small_w: int, image_h: int, image_w: int, p1_probability: float = 0.5) -> torch.Tensor
# Generates a batch of random binary masks for the RISE (Randomized Input Sampling for Explanation) method.
# Each small mask is upsampled to the image size and randomly cropped.
# From TP7 (Optional Part).

# Entrées :
# - num_masks (int): Number of random masks to generate.
# - mask_small_h (int): Height of the initial small random mask.
# - mask_small_w (int): Width of the initial small random mask.
# - image_h (int): Height of the target image (and final mask size).
# - image_w (int): Width of the target image (and final mask size).
# - p1_probability (float, optional): Probability of a '1' in the small binary mask. Defaults to 0.5.

# Sorties :
# - masks_tensor (torch.Tensor): A tensor of shape (num_masks, image_h, image_w) containing the generated masks,
#                                normalized to [0, 1].
import torch
import numpy as np
from PIL import Image # For resizing with different interpolation methods
# from skimage.transform import resize as skimage_resize # Alternative resize

def create_rise_masks(num_masks: int, mask_small_h: int, mask_small_w: int, image_h: int, image_w: int, p1_probability: float = 0.5) -> torch.Tensor:
    """
    Generates random binary masks, upsamples, and crops them for RISE.
    """
    masks_list = []
    for _ in range(num_masks):
        # 1. Create small random binary mask (0s and 1s)
        small_mask_np = np.random.choice([0, 1], size=(mask_small_h, mask_small_w), p=[1 - p1_probability, p1_probability])
        
        # 2. Upsample the small mask to be slightly larger than the image
        # Using PIL for resizing as in the original TP example (Image.BILINEAR)
        # Scale factor should make it a bit larger than image_h, image_w before cropping
        # Target upsampled size: (image_h + mask_small_h, image_w + mask_small_w) to allow for random cropping later.
        # No, the TP code uses (image_h + h, image_w + w) for resize, which is (image_h + mask_small_h, image_w + mask_small_w)
        # The variable names h,w in TP7's create_mask seem to refer to mask_small_h, mask_small_w here
        
        upsampled_h = image_h + mask_small_h 
        upsampled_w = image_w + mask_small_w

        # Convert to PIL Image, then resize
        small_mask_pil = Image.fromarray((small_mask_np * 255).astype(np.uint8), mode='L') # 'L' for grayscale
        upsampled_mask_pil = small_mask_pil.resize((upsampled_w, upsampled_h), resample=Image.Resampling.BILINEAR)
        upsampled_mask_np = np.array(upsampled_mask_pil, dtype=np.float32) / 255.0 # Back to [0,1] float

        # 3. Randomly crop the upsampled mask to the image size
        crop_y = np.random.randint(0, mask_small_h + 1) # Max y_start for crop
        crop_x = np.random.randint(0, mask_small_w + 1) # Max x_start for crop
        
        final_mask_np = upsampled_mask_np[crop_y : crop_y + image_h, crop_x : crop_x + image_w]
        
        # Normalize (though it should be close to [0,1] already from /255.0)
        min_val, max_val = final_mask_np.min(), final_mask_np.max()
        if max_val > min_val: # Avoid division by zero if mask is flat
            final_mask_np = (final_mask_np - min_val) / (max_val - min_val)
        else: # If flat, it's likely all 0s or all 1s from p1_probability being 0 or 1.
            final_mask_np = np.ones_like(final_mask_np) * min_val # Keep it flat

        masks_list.append(torch.from_numpy(final_mask_np))
        
    return torch.stack(masks_list) # Shape: (num_masks, image_h, image_w)
```

```python
## def apply_rise_masks_and_predict(model: nn.Module, original_image_tensor: torch.Tensor, rise_masks_tensor: torch.Tensor, batch_size_preds: int = 32, device_str: str = "auto") -> tuple[torch.Tensor, torch.Tensor]
# Applies a set of RISE masks to an original image, creating masked images,
# and then gets predictions from a model for these masked images.
# From TP7 (Optional Part).

# Entrées :
# - model (nn.Module): The PyTorch model to use for predictions.
# - original_image_tensor (torch.Tensor): The original input image, shape (C, H, W) or (1, C, H, W).
# - rise_masks_tensor (torch.Tensor): Tensor of RISE masks, shape (NumMasks, H, W).
# - batch_size_preds (int, optional): Batch size for running predictions to manage memory. Defaults to 32.
# - device_str (str, optional): Device ("cpu", "cuda", "auto"). Defaults to "auto".

# Sorties :
# - masked_images_tensor (torch.Tensor): Tensor of all masked images, shape (NumMasks, C, H, W).
# - predictions_tensor (torch.Tensor): Tensor of model predictions (softmax probabilities) for masked images,
#                                     shape (NumMasks, NumClasses).
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def apply_rise_masks_and_predict(
    model: nn.Module, 
    original_image_tensor: torch.Tensor, # Shape (C, H, W) or (1, C, H, W)
    rise_masks_tensor: torch.Tensor,     # Shape (NumMasks, H, W)
    batch_size_preds: int = 32, 
    device_str: str = "auto"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RISE masks to an image, gets model predictions for masked images.
    Returns all masked images and their corresponding predictions (softmax probabilities).
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model.to(device)
    model.eval()

    if original_image_tensor.ndim == 3: # (C, H, W)
        original_image_tensor = original_image_tensor.unsqueeze(0) # Add batch dim: (1, C, H, W)
    
    original_image_tensor = original_image_tensor.to(device)
    rise_masks_tensor = rise_masks_tensor.to(device)

    num_masks = rise_masks_tensor.shape[0]
    num_channels = original_image_tensor.shape[1]
    
    # Prepare to store all masked images and predictions
    #masked_images_list = [] # Can be memory intensive if num_masks is large
    all_predictions_list = []

    # For masked_images_tensor, pre-allocate if possible, or handle in batches
    # If storing all masked images is too much, this function could be a generator
    # or only return predictions. For RISE, we don't strictly need to return all masked images.
    # The TP example stores X (masked images) and masks separately.
    # Let's create masked images on the fly for prediction to save memory for X.
    
    print(f"Applying {num_masks} masks and predicting in batches of {batch_size_preds}...")
    with torch.no_grad():
        for i in tqdm(range(0, num_masks, batch_size_preds)):
            batch_masks = rise_masks_tensor[i : i + batch_size_preds] # (batch, H, W)
            
            # Element-wise multiplication: original_image (1,C,H,W) * batch_masks (batch,1,H,W)
            # Unsqueeze batch_masks to make it broadcastable with image channels
            current_batch_masked_images = original_image_tensor * batch_masks.unsqueeze(1) # (batch, C, H, W)
            
            # masked_images_list.append(current_batch_masked_images.cpu()) # Collect if needed
            
            # Get predictions for the current batch of masked images
            logits = model(current_batch_masked_images) # (batch, NumClasses)
            probabilities = F.softmax(logits, dim=1)    # (batch, NumClasses)
            all_predictions_list.append(probabilities.cpu())

    #all_masked_images_tensor = torch.cat(masked_images_list, dim=0) if masked_images_list else torch.empty(0)
    all_predictions_tensor = torch.cat(all_predictions_list, dim=0)
    
    # For RISE, we typically need the original masks and the predictions.
    # The masked images themselves are intermediate.
    # The TP7 optional part stores X (masked images) and masks.
    # Let's refine to match that more closely: return predictions. The masks are input.
    
    # For the purpose of the next step (create_rise_saliency_map), we need `predictions_tensor`
    # and the `rise_masks_tensor` (which was an input).
    
    # The TP7 code stores X (masked images). If this is critical, we need to manage memory.
    # For now, this function will focus on getting the predictions.
    # If X is needed, it should be generated and passed along, or generated here.
    # The function in TP7 `create_sum_mask` uses `masks` and `preds_masked`.
    # So this function should return `preds_masked`. The `masks` are already available.

    return all_predictions_tensor # Shape (NumMasks, NumClasses)

```

```python
## def create_rise_saliency_map(rise_masks_tensor: torch.Tensor, predictions_tensor: torch.Tensor, class_index: int) -> torch.Tensor
# Creates a RISE saliency map for a specific class by performing a weighted sum of the RISE masks.
# The weights are the model's prediction scores (probabilities) for the specified class for each masked image.
# From TP7 (Optional Part).

# Entrées :
# - rise_masks_tensor (torch.Tensor): Tensor of RISE masks, shape (NumMasks, H, W).
# - predictions_tensor (torch.Tensor): Tensor of model predictions (softmax probabilities) for masked images,
#                                     shape (NumMasks, NumClasses). This is output from
#                                     `apply_rise_masks_and_predict`.
# - class_index (int): The index of the target class for which to generate the saliency map.

# Sorties :
# - saliency_map (torch.Tensor): The generated saliency map, shape (H, W), normalized to [0, 1].

import torch

def create_rise_saliency_map(
    rise_masks_tensor: torch.Tensor,     # (NumMasks, H, W)
    predictions_tensor: torch.Tensor,  # (NumMasks, NumClasses)
    class_index: int
) -> torch.Tensor:
    """
    Creates a RISE saliency map for a specific class.
    It's a weighted sum of masks, where weights are prediction scores for that class.
    """
    if class_index < 0 or class_index >= predictions_tensor.shape[1]:
        raise ValueError(f"Invalid class_index {class_index}. Must be between 0 and {predictions_tensor.shape[1]-1}.")

    # Get the prediction scores for the target class_index for all masked images
    # These scores will be the weights for the masks
    weights_for_masks = predictions_tensor[:, class_index] # Shape: (NumMasks)

    # Ensure rise_masks_tensor and weights_for_masks are on the same device (e.g., CPU for this)
    # And ensure they are float for multiplication
    masks = rise_masks_tensor.cpu().float()
    weights = weights_for_masks.cpu().float()

    # Weighted sum of masks: Sum_i ( mask_i * weight_i )
    # masks is (NumMasks, H, W), weights is (NumMasks)
    # We need to reshape weights to (NumMasks, 1, 1) to broadcast with masks
    
    # Element-wise multiplication and sum over the NumMasks dimension
    # saliency_map = torch.sum(masks * weights.view(-1, 1, 1), dim=0)
    # Alternative using einsum for clarity (or matmul if preferred)
    saliency_map = torch.einsum('n h w, n -> h w', masks, weights)


    # Normalize the saliency map to be in [0, 1] for visualization
    min_val = torch.min(saliency_map)
    max_val = torch.max(saliency_map)
    
    if max_val > min_val: # Avoid division by zero if map is flat
        saliency_map_normalized = (saliency_map - min_val) / (max_val - min_val)
    else: # If map is flat (e.g., all zeros if all weights were zero)
        saliency_map_normalized = torch.zeros_like(saliency_map) # Or ones_like * min_val

    return saliency_map_normalized # Shape: (H, W)

```

```python
## def plot_rise_saliency_results(original_image_pil: Image.Image, class_saliency_maps: dict, main_title: str = "RISE Saliency Maps")
# Plots the original image alongside RISE saliency maps for multiple classes.
# Each saliency map is overlaid on a grayscale version of the original image.
# From TP7 (Optional Part).

# Entrées :
# - original_image_pil (PIL.Image.Image): The original input image as a PIL Image object.
# - class_saliency_maps (dict): A dictionary where keys are class names (str) and
#                               values are the corresponding saliency maps (torch.Tensor, shape (H,W)).
# - main_title (str, optional): Overall title for the figure. Defaults to "RISE Saliency Maps".

# Sorties :
# - None (displays the plot).
import matplotlib.pyplot as plt
from PIL import Image # For image operations
import numpy as np # For converting PIL to numpy for grayscale

def plot_rise_saliency_results(original_image_pil: Image.Image, class_saliency_maps: dict, main_title: str = "RISE Saliency Maps"):
    """
    Plots original image and RISE saliency map overlays for multiple classes.
    original_image_pil: PIL Image of the original input.
    class_saliency_maps: Dict {'class_name': saliency_map_tensor_HW, ...}
    """
    num_classes_to_plot = len(class_saliency_maps)
    
    if num_classes_to_plot == 0:
        print("No saliency maps provided to plot.")
        return

    fig, axes = plt.subplots(1, 1 + num_classes_to_plot, figsize=((1 + num_classes_to_plot) * 4, 4))
    fig.suptitle(main_title, fontsize=16)

    # 1. Display original image
    axes[0].imshow(original_image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Convert original PIL image to grayscale numpy array for overlay background
    original_image_gray_np = np.array(original_image_pil.convert('L'), dtype=np.float32) / 255.0 # Grayscale [0,1]

    # 2. Display saliency map overlays for each class
    plot_idx = 1
    for class_name, saliency_map_tensor in class_saliency_maps.items():
        saliency_map_np = saliency_map_tensor.cpu().numpy() # Ensure it's a NumPy array (H, W)
        
        ax = axes[plot_idx]
        ax.imshow(original_image_gray_np, cmap='gray') # Background
        im = ax.imshow(saliency_map_np, cmap='jet', alpha=0.6) # Overlay saliency map
        ax.set_title(f"RISE for: {class_name}")
        ax.axis('off')
        # Optional: Add a colorbar for the saliency map
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 
        plot_idx += 1
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    plt.show()

```
