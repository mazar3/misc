Absolument ! Je comprends mieux la demande de concision et de séparation claire entre markdown et code.

Voici la version révisée, en français, avec une structure markdown/code distincte et des fonctions regroupées lorsque c'est pertinent.

Je vais commencer par une fonction `initialiser_environnement` qui combinera les imports et la configuration de la seed. Ensuite, je suivrai avec les fonctions de manipulation de données, de visualisation, de création de modèles, d'entraînement, etc., en essayant de généraliser autant que possible.

```python
# Exécutez cette cellule pour vous assurer que le notebook est en Python
```

## def initialiser_environnement(seed_valeur: int = 42, pytorch_deterministe: bool = True)
Cette fonction initialise l'environnement pour les travaux pratiques. Elle effectue tous les imports de librairies nécessaires et configure la graine aléatoire (seed) pour la reproductibilité.

**Entrées :**
- `seed_valeur` (int, optionnel) : La valeur de la graine pour les générateurs de nombres aléatoires. Par défaut à 42.
- `pytorch_deterministe` (bool, optionnel) : Si `True`, configure PyTorch pour utiliser des algorithmes déterministes, ce qui peut affecter les performances mais améliore la reproductibilité. Par défaut à `True`.

**Sorties :**
- Aucune (affiche un message de confirmation).

```python
import os
import urllib.request
import zipfile
import glob
import warnings # Pour ignorer certains avertissements

# Manipulation de données et calculs numériques
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2 # OpenCV pour la manipulation d'images

# Machine Learning - Général
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, precision_score, recall_score, f1_score

# Machine Learning - Modèles spécifiques
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# PyTorch - Coeur
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, random_split

# Torchvision (pour les datasets, modèles pré-entraînés, transformations d'images)
import torchvision
import torchvision.datasets as tv_datasets # Renommé pour éviter conflit avec variable 'datasets'
import torchvision.models as tv_models     # Renommé pour éviter conflit
import torchvision.transforms.v2 as tv_transforms_v2 # Pour les TPs récents
from torchvision.transforms.functional import normalize, resize, to_pil_image as F_to_pil_image # Spécifique pour XAI

# PyTorch Lightning (pour structurer le code PyTorch)
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer # seed_everything est déjà dans pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_fabric.utilities.seed import seed_everything as fabric_seed_everything # Pour versions plus récentes

# Torchinfo (pour résumer les modèles)
from torchinfo import summary as torchinfo_summary # Renommé

# Torchmetrics (pour les métriques)
from torchmetrics import Accuracy as TorchmetricsAccuracy # Renommé
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

# Weights & Biases (pour le suivi des expériences)
import wandb

# Folium (pour les cartes)
import folium
import branca.colormap

# PyCaret (pour l' AutoML)
from pycaret.regression import setup as pycaret_setup, compare_models as pycaret_compare_models, \
                                tune_model as pycaret_tune_model, evaluate_model as pycaret_evaluate_model, \
                                predict_model as pycaret_predict_model

# Librairies XAI (Explicabilité)
from torchcam.methods import GradCAM, LayerCAM
from torchcam.utils import overlay_mask as torchcam_overlay_mask # Renommé
from captum.attr import GradientShap, Occlusion, IntegratedGradients, DeepLift
from captum.attr import visualization as captum_viz
# Pour TIS (Transformer Input Sampling) - nécessite une installation/configuration spécifique
# from Transformer_Input_Sampling.tis import TIS
import timm # Souvent utilisé avec les Transformers ou modèles d'images avancés
# import fast_pytorch_kmeans # Si utilisé par TIS
from skimage.transform import resize as skimage_resize # Pour l'exemple RISE potentiellement
from einops import rearrange # Souvent utilisé avec les Transformers

# Pillow (PIL) - Manipulation d'images
from PIL import Image

# Utilitaires IPython
from IPython.display import display

def initialiser_environnement(seed_valeur: int = 42, pytorch_deterministe: bool = True):
    """
    Initialise les imports globaux et la graine aléatoire pour la reproductibilité.
    """
    pl.seed_everything(seed_valeur, workers=True) # workers=True est souvent par défaut ou recommandé

    if pytorch_deterministe:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Certaines versions de PyTorch nécessitent ceci pour une déterminisme complet sur GPU :
        # torch.use_deterministic_algorithms(True)
        # Et parfois une variable d'environnement pour CUBLAS :
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # ou ":16:8"

    # Ignorer les avertissements spécifiques si nécessaire, par exemple :
    warnings.filterwarnings('ignore', category=UserWarning, message='The given NumPy array is not writable')
    # Désactiver les avertissements de PyTorch Lightning sur les versions de packages
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists and is not empty.*")


    print(f"Environnement initialisé. Graine aléatoire (seed) configurée à {seed_valeur}.")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU non disponible, utilisation du CPU.")

# Appel pour initialiser lors de l'exécution du notebook
# initialiser_environnement()
```

## def creer_dataframe_simple(dictionnaire_donnees: dict)
Crée un DataFrame Pandas à partir d'un dictionnaire Python.
Utilisé dans le TP1 pour une introduction basique à Pandas.

**Entrées :**
- `dictionnaire_donnees` (dict) : Un dictionnaire où les clés sont les noms des colonnes et les valeurs sont des listes représentant les données des colonnes.

**Sorties :**
- `df` (pd.DataFrame) : Le DataFrame Pandas créé à partir du dictionnaire.

```python
def creer_dataframe_simple(dictionnaire_donnees: dict) -> pd.DataFrame:
    """
    Crée un DataFrame Pandas à partir d'un dictionnaire Python.
    Exemple: donnees = {"col1": [1, 2], "col2": [3, 4]}
    """
    df = pd.DataFrame(dictionnaire_donnees)
    return df
```

## def trier_dataframe(df: pd.DataFrame, colonnes_tri: list, ordres_tri: list)
Trie un DataFrame Pandas selon une ou plusieurs colonnes, chacune avec un ordre de tri spécifié.
Utilisé dans le TP1 pour l'exploration des données.

**Entrées :**
- `df` (pd.DataFrame) : Le DataFrame à trier.
- `colonnes_tri` (list) : Une liste de noms de colonnes pour le tri.
- `ordres_tri` (list) : Une liste de booléens correspondant à `colonnes_tri`,
où `True` signifie un tri ascendant et `False` un tri descendant.

**Sorties :**
- `df_trie` (pd.DataFrame) : Le DataFrame trié.

```python
def trier_dataframe(df: pd.DataFrame, colonnes_tri: list, ordres_tri: list) -> pd.DataFrame:
    """
    Trie un DataFrame Pandas selon les colonnes et ordres spécifiés.
    Exemple: trier_dataframe(df, colonnes_tri=["population", "superficie_km2"], ordres_tri=[False, False])
    """
    if len(colonnes_tri) != len(ordres_tri):
        raise ValueError("La longueur de colonnes_tri et ordres_tri doit être la même.")
    df_trie = df.sort_values(by=colonnes_tri, ascending=ordres_tri)
    return df_trie
```

## def visualiser_barres_seaborn(df: pd.DataFrame, col_x: str, col_y: str, titre: str, label_x: str, label_y: str, palette: str = "Blues_r", col_couleur_hue: str = None, taille_figure: tuple = (8, 5), orientation: str = 'h')
Crée un diagramme en barres (horizontal ou vertical) en utilisant Seaborn.
Utilisé dans le TP1 pour visualiser les populations/superficies des villes.

**Entrées :**
- `df` (pd.DataFrame) : Le DataFrame contenant les données.
- `col_x` (str) : Le nom de la colonne pour l'axe des x (ou l'axe des valeurs pour un diagramme horizontal).
- `col_y` (str) : Le nom de la colonne pour l'axe des y (ou l'axe des catégories pour un diagramme horizontal).
- `titre` (str) : Le titre du graphique.
- `label_x` (str) : L'étiquette pour l'axe des x.
- `label_y` (str) : L'étiquette pour l'axe des y.
- `palette` (str, optionnel) : Palette de couleurs Seaborn. Par défaut "Blues_r".
- `col_couleur_hue` (str, optionnel) : Nom de la colonne pour l'encodage des couleurs (hue). Par défaut `None`.
- `taille_figure` (tuple, optionnel) : Taille de la figure. Par défaut (8, 5).
- `orientation` (str, optionnel) : Orientation du diagramme en barres, 'h' pour horizontal, 'v' pour vertical. Par défaut 'h'.

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_barres_seaborn(df: pd.DataFrame, col_x: str, col_y: str, titre: str, label_x: str, label_y: str, palette: str = "Blues_r", col_couleur_hue: str = None, taille_figure: tuple = (8, 5), orientation: str = 'h'):
    """
    Crée un diagramme en barres avec Seaborn.
    Pour un diagramme horizontal (orientation='h'): col_y est catégorique, col_x est numérique.
    Pour un diagramme vertical (orientation='v'): col_x est catégorique, col_y est numérique.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=taille_figure)
    if orientation == 'h':
        sns.barplot(x=col_x, y=col_y, data=df, palette=palette, hue=col_couleur_hue, orient='h')
    elif orientation == 'v':
        sns.barplot(x=col_x, y=col_y, data=df, palette=palette, hue=col_couleur_hue, orient='v')
    else:
        raise ValueError("L'orientation doit être 'h' ou 'v'")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(titre)
    plt.show()
```

## def pretraiter_caracteristiques_housing_californie(dataframe_housing: pd.DataFrame)
Prétraite les caractéristiques pour le jeu de données des logements en Californie (utilisé dans TP1/TP2).
Sélectionne des caractéristiques spécifiques et crée une caractéristique synthétique "rooms_per_person".

**Entrées :**
- `dataframe_housing` (pd.DataFrame) : Le DataFrame d'entrée avec les données des logements en Californie.
  Colonnes attendues : "latitude", "longitude", "housing_median_age", "total_rooms",
  "total_bedrooms", "population", "households", "median_income".

**Sorties :**
- `caracteristiques_traitees` (pd.DataFrame) : DataFrame avec les caractéristiques sélectionnées et créées.

```python
def pretraiter_caracteristiques_housing_californie(dataframe_housing: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne des caractéristiques et crée la caractéristique synthétique "rooms_per_person"
    pour le jeu de données des logements en Californie (version TP1/TP2).
    """
    caracteristiques_traitees = dataframe_housing.copy()
    colonnes_selectionnees = ["latitude", "longitude", "housing_median_age", "total_rooms",
                              "total_bedrooms", "population", "households", "median_income"]
    
    # Vérifier si les colonnes nécessaires pour la nouvelle caractéristique existent
    if "total_rooms" in dataframe_housing.columns and "population" in dataframe_housing.columns and dataframe_housing["population"].ne(0).all():
        caracteristiques_traitees["rooms_per_person"] = dataframe_housing["total_rooms"] / dataframe_housing["population"]
        colonnes_finales_selectionnees = colonnes_selectionnees + ["rooms_per_person"]
    else:
        print("Avertissement : Les colonnes 'total_rooms' ou 'population' sont manquantes ou 'population' contient des zéros. La caractéristique 'rooms_per_person' ne peut pas être créée.")
        colonnes_finales_selectionnees = colonnes_selectionnees

    # Filtrer pour ne garder que les colonnes existantes dans le DataFrame traité
    colonnes_existantes_a_selectionner = [col for col in colonnes_finales_selectionnees if col in caracteristiques_traitees.columns]
    
    return caracteristiques_traitees[colonnes_existantes_a_selectionner]
```

## def pretraiter_cibles_housing_californie(dataframe_housing: pd.DataFrame)
Prétraite la variable cible "median_house_value" pour le jeu de données des logements en Californie (utilisé dans TP1/TP2).
Met à l'échelle "median_house_value" en la divisant par 1000.

**Entrées :**
- `dataframe_housing` (pd.DataFrame) : Le DataFrame d'entrée avec les données des logements en Californie.
  Colonne attendue : "median_house_value".

**Sorties :**
- `cibles_traitees` (pd.Series) : Série Pandas contenant la "median_house_value" mise à l'échelle.

```python
def pretraiter_cibles_housing_californie(dataframe_housing: pd.DataFrame) -> pd.Series:
    """
    Met à l'échelle "median_house_value" en la divisant par 1000.0 (version TP1/TP2).
    Retourne uniquement la colonne cible sous forme de Série.
    """
    if "median_house_value" not in dataframe_housing.columns:
        raise KeyError("La colonne 'median_house_value' est introuvable dans le DataFrame.")
        
    donnees_traitees = dataframe_housing.copy()
    donnees_traitees["median_house_value"] = donnees_traitees["median_house_value"] / 1000.0
    return donnees_traitees["median_house_value"]
```

## def visualiser_donnees_housing_dispersion_carte(donnees_entrainement: pd.DataFrame, donnees_validation: pd.DataFrame)
Visualise les données d'entraînement et de validation pour les logements en Californie sur des nuages de points,
simulant une carte en utilisant la longitude et la latitude, colorée par "median_house_value".
Utilisé dans le TP1.

**Entrées :**
- `donnees_entrainement` (pd.DataFrame) : DataFrame pour l'entraînement, doit inclure "longitude", "latitude", "median_house_value".
- `donnees_validation` (pd.DataFrame) : DataFrame pour la validation, doit inclure "longitude", "latitude", "median_house_value".

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_donnees_housing_dispersion_carte(donnees_entrainement: pd.DataFrame, donnees_validation: pd.DataFrame):
  """
  Visualise les données d'entraînement et de validation des logements sur des nuages de points.
  La longitude et la latitude sont utilisées pour les axes x et y.
  Les points sont colorés par median_house_value.
  """
  plt.figure(figsize=(13, 8))

  # Graphique des données de validation
  ax1 = plt.subplot(1, 2, 1)
  ax1.set_title("Données de Validation")
  ax1.set_autoscaley_on(False)
  ax1.set_ylim([32, 43])
  ax1.set_autoscalex_on(False)
  ax1.set_xlim([-126, -112])
  # Vérifier si median_house_value existe et n'est pas nul avant la division
  if "median_house_value" in donnees_validation.columns and not donnees_validation["median_house_value"].isnull().all():
      max_val_val = donnees_validation["median_house_value"].max()
      if max_val_val == 0: max_val_val = 1 # Éviter la division par zéro si max est 0
      couleurs_val = donnees_validation["median_house_value"] / max_val_val
  else:
      couleurs_val = "blue" # Couleur par défaut si la colonne est manquante ou toutes nulles
      print("Avertissement: 'median_house_value' manquante ou nulle dans les données de validation pour la coloration.")

  scatter_val = ax1.scatter(donnees_validation["longitude"],
                            donnees_validation["latitude"],
                            cmap="coolwarm",
                            c=couleurs_val)
  ax1.set_xlabel("Longitude")
  ax1.set_ylabel("Latitude")
  if isinstance(couleurs_val, pd.Series) or isinstance(couleurs_val, np.ndarray): # Ajouter une colorbar si les couleurs sont basées sur des données
    plt.colorbar(scatter_val, ax=ax1, label="Valeur normalisée de la maison")


  # Graphique des données d'entraînement
  ax2 = plt.subplot(1,2,2)
  ax2.set_title("Données d'Entraînement")
  ax2.set_autoscaley_on(False)
  ax2.set_ylim([32, 43])
  ax2.set_autoscalex_on(False)
  ax2.set_xlim([-126, -112])
  if "median_house_value" in donnees_entrainement.columns and not donnees_entrainement["median_house_value"].isnull().all():
      max_val_train = donnees_entrainement["median_house_value"].max()
      if max_val_train == 0: max_val_train = 1
      couleurs_train = donnees_entrainement["median_house_value"] / max_val_train
  else:
      couleurs_train = "red"
      print("Avertissement: 'median_house_value' manquante ou nulle dans les données d'entraînement pour la coloration.")

  scatter_train = ax2.scatter(donnees_entrainement["longitude"],
                              donnees_entrainement["latitude"],
                              cmap="coolwarm",
                              c=couleurs_train)
  ax2.set_xlabel("Longitude")
  ax2.set_ylabel("Latitude")
  if isinstance(couleurs_train, pd.Series) or isinstance(couleurs_train, np.ndarray):
    plt.colorbar(scatter_train, ax=ax2, label="Valeur normalisée de la maison")
  
  plt.tight_layout()
  plt.show()
```

## def visualiser_carte_correlation(df: pd.DataFrame, methode: str = 'pearson', taille_figure: tuple = (12, 8), afficher_annotations: bool = True, format_annotations: str = '.2f', palette_couleurs: str = 'coolwarm')
Calcule et affiche une carte de chaleur (heatmap) des corrélations pour un DataFrame.
Utilisé dans le TP1.

**Entrées :**
- `df` (pd.DataFrame) : Le DataFrame pour lequel calculer les corrélations.
- `methode` (str, optionnel) : Méthode de corrélation ('pearson', 'kendall', 'spearman'). Par défaut 'pearson'.
- `taille_figure` (tuple, optionnel) : Taille de la figure. Par défaut (12, 8).
- `afficher_annotations` (bool, optionnel) : Si `True`, écrit la valeur des données dans chaque cellule. Par défaut `True`.
- `format_annotations` (str, optionnel) : Format de chaîne à utiliser lorsque `afficher_annotations` est `True`. Par défaut '.2f'.
- `palette_couleurs` (str, optionnel) : Nom ou objet de la palette de couleurs Matplotlib. Par défaut 'coolwarm'.

**Sorties :**
- `matrice_corr` (pd.DataFrame) : La matrice de corrélation calculée.
- Affiche la carte de chaleur.

```python
def visualiser_carte_correlation(df: pd.DataFrame, methode: str = 'pearson', taille_figure: tuple = (12, 8), afficher_annotations: bool = True, format_annotations: str = '.2f', palette_couleurs: str = 'coolwarm') -> pd.DataFrame:
    """
    Calcule et affiche une carte de chaleur des corrélations pour le DataFrame donné.
    """
    plt.figure(figsize=taille_figure)
    matrice_corr = df.corr(method=methode)
    sns.heatmap(matrice_corr, annot=afficher_annotations, fmt=format_annotations, cmap=palette_couleurs, center=0)
    plt.title(f"Matrice de Corrélation ({methode.capitalize()})")
    plt.show()
    return matrice_corr
```

## def visualiser_distribution_variable(serie_pandas: pd.Series, titre: str, label_x: str, label_y: str = "Fréquence", nb_bins: int = 30, afficher_kde: bool = True, taille_figure: tuple = (10, 6), couleur: str = 'blue')
Affiche la distribution d'une variable numérique en utilisant `histplot` de Seaborn.
Utilisé dans le TP1.

**Entrées :**
- `serie_pandas` (pd.Series) : La Série Pandas dont la distribution doit être tracée.
- `titre` (str) : Le titre du graphique.
- `label_x` (str) : L'étiquette pour l'axe des x.
- `label_y` (str, optionnel) : L'étiquette pour l'axe des y. Par défaut "Fréquence".
- `nb_bins` (int, optionnel) : Nombre de bins pour l'histogramme. Par défaut 30.
- `afficher_kde` (bool, optionnel) : S'il faut tracer une estimation de la densité du noyau (KDE). Par défaut `True`.
- `taille_figure` (tuple, optionnel) : Taille de la figure. Par défaut (10, 6).
- `couleur` (str, optionnel) : Couleur du graphique. Par défaut 'blue'.

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_distribution_variable(serie_pandas: pd.Series, titre: str, label_x: str, label_y: str = "Fréquence", nb_bins: int = 30, afficher_kde: bool = True, taille_figure: tuple = (10, 6), couleur: str = 'blue'):
    """
    Affiche la distribution d'une variable numérique en utilisant histplot de Seaborn.
    """
    plt.figure(figsize=taille_figure)
    sns.histplot(serie_pandas, kde=afficher_kde, bins=nb_bins, color=couleur)
    plt.title(titre)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.show()
```

## def visualiser_donnees_sur_carte_folium(donnees: pd.DataFrame, nom_split_donnees: str, col_latitude: str = 'latitude', col_longitude: str = 'longitude', col_valeur: str = 'median_house_value', lat_centre: float = 37.16, lon_centre: float = -120.43, zoom_initial: int = 6)
Visualise des points de données sur une carte Folium, colorés par une valeur spécifiée.
Utilisé dans le TP1.

**Entrées :**
- `donnees` (pd.DataFrame) : DataFrame contenant la latitude, la longitude et la valeur à visualiser.
- `nom_split_donnees` (str) : Nom du découpage des données (ex: "Données d'Entraînement") pour le titre.
- `col_latitude` (str, optionnel) : Nom de la colonne de latitude. Par défaut 'latitude'.
- `col_longitude` (str, optionnel) : Nom de la colonne de longitude. Par défaut 'longitude'.
- `col_valeur` (str, optionnel) : Nom de la colonne dont les valeurs détermineront la couleur des points. Par défaut 'median_house_value'.
- `lat_centre` (float, optionnel) : Latitude pour le centre de la carte. Par défaut 37.16.
- `lon_centre` (float, optionnel) : Longitude pour le centre de la carte. Par défaut -120.43.
- `zoom_initial` (int, optionnel) : Niveau de zoom initial pour la carte. Par défaut 6.

**Sorties :**
- `m` (folium.Map) : L'objet carte Folium.

```python
def visualiser_donnees_sur_carte_folium(donnees: pd.DataFrame, nom_split_donnees: str, col_latitude: str = 'latitude', col_longitude: str = 'longitude', col_valeur: str = 'median_house_value', lat_centre: float = 37.16, lon_centre: float = -120.43, zoom_initial: int = 6) -> folium.Map:
    """
    Visualise des points de données sur une carte Folium, colorés par une valeur spécifiée.
    """
    m = folium.Map(location=[lat_centre, lon_centre], tiles="OpenStreetMap", zoom_start=zoom_initial)

    # Définir une palette de couleurs basée sur la valeur médiane des maisons
    # S'assurer que col_valeur existe et est numérique
    if col_valeur not in donnees.columns or not pd.api.types.is_numeric_dtype(donnees[col_valeur]):
        print(f"Avertissement : Colonne '{col_valeur}' introuvable ou non numérique. Utilisation d'une couleur/rayon par défaut.")
        utiliser_palette = False
    else:
        utiliser_palette = True
        val_min = donnees[col_valeur].min()
        val_max = donnees[col_valeur].max()
        if val_min == val_max: # Éviter la division par zéro si toutes les valeurs sont identiques
             palette_couleurs = branca.colormap.LinearColormap(colors=['blue', 'blue'], index=[0,1], vmin=0, vmax=1)
        else:
            # Création de l'index pour la colormap. Par exemple, utiliser min, median, max.
            # Ou simplement min et max comme dans le code original du TP.
            # index_palette = [val_min, (val_min + val_max) / 2, val_max]
            # couleurs_palette = ['blue', 'yellow', 'red']
            # palette_couleurs = branca.colormap.LinearColormap(colors=couleurs_palette, index=index_palette, vmin=val_min, vmax=val_max)
            palette_couleurs = branca.colormap.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=val_min, vmax=val_max)


    for i in range(len(donnees)):
        couleur_point = 'blue' # Couleur par défaut
        opacite_remplissage = 0.6
        rayon_cercle = 50 # Peut aussi être basé sur les données

        if utiliser_palette:
            valeur_actuelle = donnees.iloc[i][col_valeur]
            couleur_point = palette_couleurs(valeur_actuelle)

        folium.Circle(
            location=[donnees.iloc[i][col_latitude], donnees.iloc[i][col_longitude]],
            radius=rayon_cercle,
            color=couleur_point,
            fill=True,
            fill_color=couleur_point, # Utiliser la même couleur pour le remplissage
            fill_opacity=opacite_remplissage,
            tooltip=f"{col_valeur}: {donnees.iloc[i].get(col_valeur, 'N/A')}" # Afficher la valeur au survol
        ).add_to(m)

    titre_html = f'<h3 align="center" style="font-size:16px"><b>{nom_split_donnees}</b></h3>'
    m.get_root().html.add_child(folium.Element(titre_html))
    if utiliser_palette: # Ajouter la légende de la palette de couleurs seulement si elle est utilisée
        m.add_child(palette_couleurs)
    return m
```

## class ModeleLineaireSimplePyTorch(nn.Module)
Un modèle de régression linéaire simple (y = ax + b) utilisant `nn.Module` de PyTorch.
Utilisé dans le TP2 pour une introduction à la définition de modèles PyTorch.

**Entrées (pour `__init__`) :**
- `dim_entree` (int) : Dimensionalité des caractéristiques d'entrée (typiquement 1 pour une régression linéaire simple).
- `dim_sortie` (int) : Dimensionalité de la sortie (typiquement 1 pour une régression linéaire simple).

**Entrées (pour `forward`) :**
- `x` (torch.Tensor) : Le tenseur d'entrée.

**Sorties (de `forward`) :**
- (torch.Tensor) : Le tenseur de sortie (prédictions).

```python
class ModeleLineaireSimplePyTorch(nn.Module):
    """
    Un modèle de régression linéaire simple (y = ax + b) utilisant PyTorch.
    dim_entree: nombre de caractéristiques d'entrée (ex: 1 pour X)
    dim_sortie: nombre de valeurs de sortie (ex: 1 pour Y)
    """
    def __init__(self, dim_entree: int = 1, dim_sortie: int = 1):
        super(ModeleLineaireSimplePyTorch, self).__init__()
        self.linear = nn.Linear(dim_entree, dim_sortie)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

## def entrainer_modele_lineaire_simple_pytorch(modele: nn.Module, critere: nn.Module, optimiseur: optim.Optimizer, X_tenseur: torch.Tensor, Y_tenseur: torch.Tensor, nb_epochs: int = 100, afficher_chaque: int = 10, tracer_chaque: int = 0)
Entraîne un modèle PyTorch simple (comme `ModeleLineaireSimplePyTorch`).
Gère la boucle d'entraînement : passe avant, calcul de la perte, passe arrière, mise à jour de l'optimiseur.
Utilisé dans le TP2.

**Entrées :**
- `modele` (nn.Module) : Le modèle PyTorch à entraîner.
- `critere` (nn.Module) : La fonction de perte (ex: `nn.MSELoss()`).
- `optimiseur` (optim.Optimizer) : L'optimiseur (ex: `torch.optim.SGD(...)`).
- `X_tenseur` (torch.Tensor) : Le tenseur de données d'entrée.
- `Y_tenseur` (torch.Tensor) : Le tenseur de données cibles.
- `nb_epochs` (int, optionnel) : Nombre d'époques d'entraînement. Par défaut 100.
- `afficher_chaque` (int, optionnel) : Fréquence d'affichage de la perte (ex: toutes les 10 époques). Si 0, pas d'affichage.
- `tracer_chaque` (int, optionnel) : Fréquence de traçage des prédictions vs original. Si 0, pas de traçage pendant l'entraînement.

**Sorties :**
- `pertes` (list) : Une liste des valeurs de perte, une pour chaque époque.
- Affiche les informations de perte et des graphiques optionnels pendant l'entraînement.

```python
def entrainer_modele_lineaire_simple_pytorch(modele: nn.Module, critere: nn.Module, optimiseur: optim.Optimizer, X_tenseur: torch.Tensor, Y_tenseur: torch.Tensor, nb_epochs: int = 100, afficher_chaque: int = 10, tracer_chaque: int = 0) -> list:
    """
    Entraîne un modèle PyTorch simple.
    `tracer_chaque > 0` tracera les prédictions en supposant que X_tenseur et Y_tenseur sont 1D pour le nuage de points.
    """
    pertes = []
    
    # S'assurer que X_tenseur et Y_tenseur sont des tenseurs float
    if not isinstance(X_tenseur, torch.FloatTensor):
        X_tenseur = X_tenseur.float()
    if not isinstance(Y_tenseur, torch.FloatTensor):
        Y_tenseur = Y_tenseur.float()

    # Redimensionner s'ils sont 1D pour la compatibilité avec nn.Linear
    if X_tenseur.ndim == 1:
        X_tenseur = X_tenseur.view(-1, 1)
    if Y_tenseur.ndim == 1:
        Y_tenseur = Y_tenseur.view(-1, 1)

    for epoch in range(nb_epochs):
        # 1. Réinitialiser les gradients
        optimiseur.zero_grad()

        # 2. Passe avant
        predictions = modele(X_tenseur)

        # 3. Calculer la perte
        perte = critere(predictions, Y_tenseur)
        pertes.append(perte.item())

        # 4. Passe arrière
        perte.backward()

        # 5. Mettre à jour les poids
        optimiseur.step()

        if afficher_chaque > 0 and (epoch + 1) % afficher_chaque == 0:
            print(f"Époque {epoch + 1}/{nb_epochs}, Perte : {perte.item():.4f}")

        if tracer_chaque > 0 and (epoch + 1) % tracer_chaque == 0 :
            plt.figure(figsize=(6, 4))
            # Détacher les tenseurs et convertir en numpy pour le traçage
            # En supposant que X_tenseur était initialement 1D pour ce graphique spécifique
            X_original_pour_graph = X_tenseur.view(-1).detach().cpu().numpy()
            Y_original_pour_graph = Y_tenseur.view(-1).detach().cpu().numpy()
            Y_predit_pour_graph = predictions.view(-1).detach().cpu().numpy()

            plt.scatter(X_original_pour_graph, Y_original_pour_graph, color='blue', label='Données Originales', s=10)
            plt.scatter(X_original_pour_graph, Y_predit_pour_graph, color='red', label='Prédictions', s=10, alpha=0.6)
            # plt.plot(X_original_pour_graph, Y_predit_pour_graph, color='red', linestyle='--', alpha=0.5) # Ligne de tendance
            # Pour tracer la ligne apprise, il faut trier X si ce n'est pas déjà le cas
            indices_tries = np.argsort(X_original_pour_graph)
            plt.plot(X_original_pour_graph[indices_tries], Y_predit_pour_graph[indices_tries], color='red', linestyle='-', alpha=0.7, label='Ligne Apprise')


            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Époque {epoch+1} Prédictions')
            plt.legend()
            plt.show()
            
    return pertes
```

## def mettre_a_echelle_donnees_minmax(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None, X_val: pd.DataFrame = None, y_val: pd.Series = None)
Met à l'échelle les caractéristiques (X) et la cible (y) en utilisant `MinMaxScaler`.
Ajuste le scaler sur les données d'entraînement et transforme les ensembles d'entraînement, de validation et de test.
La variable cible `y` est redimensionnée en 2D avant la mise à l'échelle, car `MinMaxScaler` attend une entrée 2D.
Utilisé dans le TP2 pour les données de logement et adapté dans le TP3 (comme `normalize_data`).

**Entrées :**
- `X_train` (pd.DataFrame) : Caractéristiques d'entraînement.
- `y_train` (pd.Series) : Cible d'entraînement.
- `X_test` (pd.DataFrame, optionnel) : Caractéristiques de test.
- `y_test` (pd.Series, optionnel) : Cible de test.
- `X_val` (pd.DataFrame, optionnel) : Caractéristiques de validation.
- `y_val` (pd.Series, optionnel) : Cible de validation.

**Sorties :**
- `tuple` : Contient les données mises à l'échelle (X_train_scaled, y_train_scaled, ...),
et les scalers ajustés (scaler_X, scaler_y).
Ordre : X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, scaler_X, scaler_y.
Si les données de validation/test ne sont pas fournies, les sorties correspondantes seront `None`.

```python
def mettre_a_echelle_donnees_minmax(X_train: pd.DataFrame, y_train: pd.Series, 
                                   X_test: pd.DataFrame = None, y_test: pd.Series = None, 
                                   X_val: pd.DataFrame = None, y_val: pd.Series = None):
    """
    Met à l'échelle les caractéristiques (X) et la cible (y) en utilisant MinMaxScaler.
    S'ajuste sur les données d'entraînement et transforme les ensembles d'entraînement, de validation et de test.
    y est redimensionné en 2D pour la compatibilité avec le scaler.
    Retourne les données mises à l'échelle et les scalers ajustés.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Mettre à l'échelle les caractéristiques
    X_train_scaled_np = scaler_X.fit_transform(X_train)
    X_val_scaled_np = scaler_X.transform(X_val) if X_val is not None else None
    X_test_scaled_np = scaler_X.transform(X_test) if X_test is not None else None

    # Redimensionner y en tableau 2D car MinMaxScaler attend une entrée 2D
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

    # Reconvertir X mis à l'échelle en DataFrames si les originaux étaient des DataFrames, en conservant les colonnes et l'index
    X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index) if X_val is not None else None
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index) if X_test is not None else None
        
    # y_scaled sont des tableaux numpy, ce qui est typique pour l'entrée/sortie du modèle

    return X_train_scaled_df, y_train_scaled, X_val_scaled_df, y_val_scaled, X_test_scaled_df, y_test_scaled, scaler_X, scaler_y
```

## def calculer_metriques_regression(y_vrai_echelle: np.ndarray, y_pred_echelle: np.ndarray, scaler_y: MinMaxScaler, epsilon: float = 1e-7)
Calcule les scores MAE, MAPE et R2 pour les tâches de régression.
Important : elle transforme d'abord inversement les valeurs vraies et prédites mises à l'échelle
en utilisant le `scaler_y` fourni avant de calculer les métriques, afin que les métriques soient dans l'échelle d'origine.
Utilisé dans le TP2.

**Entrées :**
- `y_vrai_echelle` (np.ndarray) : Valeurs cibles vraies mises à l'échelle (typiquement 2D, ex: issues de `MinMaxScaler`).
- `y_pred_echelle` (np.ndarray) : Valeurs cibles prédites mises à l'échelle (typiquement 2D).
- `scaler_y` (MinMaxScaler) : L'objet scaler qui a été utilisé pour mettre à l'échelle `y_vrai_echelle`.
- `epsilon` (float, optionnel) : Petite valeur à ajouter au dénominateur pour MAPE afin d'éviter la division par zéro.

**Sorties :**
- `dict` : Un dictionnaire contenant 'mae', 'mape', et 'r2'.

```python
def calculer_metriques_regression(y_vrai_echelle: np.ndarray, y_pred_echelle: np.ndarray, scaler_y: MinMaxScaler, epsilon: float = 1e-7) -> dict:
    """
    Calcule MAE, MAPE et R2 après transformation inverse des prédictions mises à l'échelle.
    Suppose que y_vrai_echelle et y_pred_echelle sont des tableaux numpy 2D (sortie de MinMaxScaler).
    """
    # S'assurer que les entrées sont des tableaux numpy et redimensionner si elles sont 1D
    y_vrai_s = np.asarray(y_vrai_echelle)
    y_pred_s = np.asarray(y_pred_echelle)

    if y_vrai_s.ndim == 1:
        y_vrai_s = y_vrai_s.reshape(-1, 1)
    if y_pred_s.ndim == 1:
        y_pred_s = y_pred_s.reshape(-1, 1)

    # Transformation inverse à l'échelle d'origine
    y_vrai_original = scaler_y.inverse_transform(y_vrai_s)
    y_pred_original = scaler_y.inverse_transform(y_pred_s)

    # Calculer MAE
    mae = mean_absolute_error(y_vrai_original, y_pred_original)

    # Calculer MAPE
    diff = np.abs(y_vrai_original - y_pred_original)
    denominateur = np.maximum(np.abs(y_vrai_original), epsilon) 
    mape = np.mean(diff / denominateur) * 100 # En pourcentage

    # Calculer R2 Score
    r2 = r2_score(y_vrai_original, y_pred_original)

    return {"mae": mae, "mape": mape, "r2": r2}
```

## def visualiser_predictions_regression(y_vrai_original: pd.Series, y_pred_original: np.ndarray, nb_echantillons_a_tracer: int = 100, titre: str = "Prédictions vs Valeurs Réelles", label_x: str = "Index Maison", label_y: str = "Valeur Maison")
Trace les valeurs vraies vs. prédites pour un modèle de régression.
Suppose que les entrées sont déjà dans leur forme originale, non mise à l'échelle.
Utilisé dans le TP2.

**Entrées :**
- `y_vrai_original` (pd.Series ou np.ndarray) : Valeurs cibles vraies à l'échelle d'origine.
- `y_pred_original` (np.ndarray) : Valeurs cibles prédites à l'échelle d'origine.
- `nb_echantillons_a_tracer` (int, optionnel) : Nombre d'échantillons initiaux à tracer. Par défaut 100.
- `titre` (str, optionnel) : Titre du graphique.
- `label_x` (str, optionnel) : Étiquette pour l'axe des x.
- `label_y` (str, optionnel) : Étiquette pour l'axe des y.

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_predictions_regression(y_vrai_original: pd.Series, y_pred_original: np.ndarray, nb_echantillons_a_tracer: int = 100, titre: str = "Prédictions vs Valeurs Réelles", label_x: str = "Index Maison", label_y: str = "Valeur Maison"):
    """
    Trace les valeurs vraies vs. prédites pour un modèle de régression.
    Les entrées y_vrai et y_pred doivent être dans leur forme originale (non mise à l'échelle).
    y_vrai_original peut être une Série Pandas ou un tableau NumPy 1D.
    y_pred_original doit être un tableau NumPy 1D.
    """
    if isinstance(y_vrai_original, pd.Series):
        y_vrai_pour_graph = y_vrai_original.values[:nb_echantillons_a_tracer]
    else:
        y_vrai_pour_graph = np.asarray(y_vrai_original)[:nb_echantillons_a_tracer]
        
    y_pred_pour_graph = np.asarray(y_pred_original).flatten()[:nb_echantillons_a_tracer] # S'assurer qu'il est 1D

    # S'assurer que les longueurs correspondent pour la création du DataFrame
    nb_echantillons_reel = min(len(y_vrai_pour_graph), len(y_pred_pour_graph))

    df_resultat = pd.DataFrame({
        'Index': list(range(nb_echantillons_reel)),
        'Valeurs Réelles': y_vrai_pour_graph[:nb_echantillons_reel],
        'Prédictions': y_pred_pour_graph[:nb_echantillons_reel]
    })

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=df_resultat, x='Index', y='Valeurs Réelles', marker='o', label='Valeurs Réelles', linestyle='-')
    sns.lineplot(data=df_resultat, x='Index', y='Prédictions', marker='x', label='Prédictions', linestyle='--')

    plt.title(titre)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.show()
```

## def pretraiter_donnees_housing_kc(dataframe_housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
Prétraite les caractéristiques et extrait la cible pour un autre jeu de données de logements (KC Housing Data - TP3).
Crée une caractéristique 'age' à partir de 'sales_yr' (dérivée de 'date') et 'yr_built'.
Sélectionne une liste spécifique de caractéristiques.

**Entrées :**
- `dataframe_housing` (pd.DataFrame) : Le DataFrame d'entrée (ex: kc_house_data.csv).
  Colonnes attendues : 'date', 'yr_built', et autres caractéristiques listées dans `selected_features`.
  Colonne cible : 'price'.

**Sorties :**
- `X` (pd.DataFrame) : DataFrame avec les caractéristiques sélectionnées et créées.
- `Y` (pd.Series) : Série contenant la cible 'price'.

```python
def pretraiter_donnees_housing_kc(dataframe_housing: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prétraite les caractéristiques pour le jeu de données KC housing (TP3).
    Ajoute la colonne 'age', sélectionne les caractéristiques et sépare la cible 'price'.
    """
    copie_housing = dataframe_housing.copy()
    
    # S'assurer que la colonne 'date' est au format datetime
    if 'date' in copie_housing.columns:
        try: # Gérer les erreurs de conversion de date potentielles
            copie_housing['sales_yr'] = pd.to_datetime(copie_housing['date']).dt.year
        except Exception as e:
            print(f"Erreur lors de la conversion de la colonne 'date': {e}. 'sales_yr' et 'age' ne seront pas créées.")
            copie_housing['sales_yr'] = None # ou une autre valeur par défaut / gestion d'erreur

        # Ajouter l'âge des bâtiments lors de la vente comme nouvelle colonne
        if 'yr_built' in copie_housing.columns and copie_housing['sales_yr'] is not None :
            copie_housing['age'] = copie_housing['sales_yr'] - copie_housing['yr_built']
        elif 'yr_built' not in copie_housing.columns:
            print("Avertissement : Colonne 'yr_built' introuvable, 'age' ne peut pas être calculé.")
        # Si sales_yr est None, age ne sera pas calculé correctement non plus
    else:
        if 'age' not in copie_housing.columns: # Si 'age' n'est pas déjà là
            print("Avertissement : Colonne 'date' introuvable et 'age' non préexistante. La caractéristique 'age' sera manquante.")

    # Liste des caractéristiques à sélectionner
    liste_caracteristiques_selectionnees = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 
        'long', 'sqft_living15', 'sqft_lot15'
    ]
    # Ajouter 'age' seulement si elle a été créée avec succès
    if 'age' in copie_housing.columns and pd.api.types.is_numeric_dtype(copie_housing['age']):
        liste_caracteristiques_selectionnees.append('age')

    # Filtrer la liste pour ne garder que les colonnes qui existent réellement dans copie_housing
    caracteristiques_finales_selectionnees = [col for col in liste_caracteristiques_selectionnees if col in copie_housing.columns]
    
    X = copie_housing[caracteristiques_finales_selectionnees]
    
    if 'price' in copie_housing.columns:
        Y = copie_housing['price']
    else:
        raise KeyError("Colonne 'price' (cible) introuvable dans le DataFrame.")
        
    return X, Y
```

## class DatasetTabulaire(Dataset)
Une classe `Dataset` PyTorch pour les données tabulaires.
Convertit des tranches de DataFrame/Série Pandas en tenseurs PyTorch.
Utilisé dans le TP3.

**Entrées (`__init__`) :**
- `donnees` (np.ndarray ou pd.DataFrame) : Les données des caractéristiques.
- `etiquettes` (np.ndarray ou pd.Series) : Les données des étiquettes.

**Entrées (`__getitem__`) :**
- `idx` (int) : Index de l'échantillon de données à récupérer.

**Sorties (`__getitem__`) :**
- `x` (torch.Tensor) : Tenseur des caractéristiques pour l'échantillon.
- `y` (torch.Tensor) : Tenseur des étiquettes pour l'échantillon.

```python
class DatasetTabulaire(Dataset):
    """
    Dataset PyTorch pour données tabulaires.
    Suppose que les données et étiquettes sont des tableaux NumPy ou peuvent être converties.
    """
    def __init__(self, donnees, etiquettes):
        # Convertir en tableau numpy si DataFrame/Série pandas
        if isinstance(donnees, pd.DataFrame):
            self.donnees = donnees.values.astype(np.float32)
        else:
            self.donnees = np.asarray(donnees, dtype=np.float32)
            
        if isinstance(etiquettes, pd.Series):
            self.etiquettes = etiquettes.values.astype(np.float32)
        else:
            self.etiquettes = np.asarray(etiquettes, dtype=np.float32)

        # S'assurer que les étiquettes sont 2D si elles ne le sont pas déjà (ex: pour les tâches de régression avec MSELoss)
        if self.etiquettes.ndim == 1:
            self.etiquettes = self.etiquettes.reshape(-1, 1)

    def __len__(self):
        return len(self.donnees)

    def __getitem__(self, idx):
        x = torch.tensor(self.donnees[idx], dtype=torch.float32)
        y = torch.tensor(self.etiquettes[idx], dtype=torch.float32) # y doit aussi être float pour la plupart des pertes de régression
        return x, y
```

## def calculer_metriques_regression_pytorch(y_vrai_tenseur: torch.Tensor, y_pred_tenseur: torch.Tensor, scaler_y: MinMaxScaler) -> dict
Calcule les scores MAE et R2 pour les tâches de régression en utilisant des tenseurs PyTorch.
Convertit les tenseurs en tableaux NumPy et effectue une transformation inverse en utilisant `scaler_y` avant le calcul des métriques.
Utilisé dans le TP3 (BaseModel).

**Entrées :**
- `y_vrai_tenseur` (torch.Tensor) : Valeurs cibles vraies (mises à l'échelle).
- `y_pred_tenseur` (torch.Tensor) : Valeurs cibles prédites (mises à l'échelle).
- `scaler_y` (MinMaxScaler) : L'objet scaler pour la transformation inverse.

**Sorties :**
- `dict` : Un dictionnaire contenant 'mae' et 'r2'.

```python
def calculer_metriques_regression_pytorch(y_vrai_tenseur: torch.Tensor, y_pred_tenseur: torch.Tensor, scaler_y: MinMaxScaler) -> dict:
    """
    Calcule MAE et R2 à partir de tenseurs PyTorch après mise à l'échelle inverse.
    """
    # Détacher les tenseurs du graphe, déplacer vers CPU, convertir en NumPy
    y_vrai_np = y_vrai_tenseur.detach().cpu().numpy()
    y_pred_np = y_pred_tenseur.detach().cpu().numpy()

    # Redimensionner s'ils ne sont pas 2D (MinMaxScaler attend du 2D)
    if y_vrai_np.ndim == 1:
        y_vrai_np = y_vrai_np.reshape(-1, 1)
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np.reshape(-1, 1)

    # Transformation inverse
    y_vrai_original = scaler_y.inverse_transform(y_vrai_np)
    y_pred_original = scaler_y.inverse_transform(y_pred_np)

    # Calculer les métriques
    mae = mean_absolute_error(y_vrai_original, y_pred_original)
    r2 = r2_score(y_vrai_original, y_pred_original)

    return {"mae": mae, "r2": r2}
```

## class ModeleLightningBaseRegression(pl.LightningModule)
Un module PyTorch Lightning de base pour les tâches de régression.
Implémente `training_step`, `validation_step`, `test_step`, `predict_step`, et `configure_optimizers`.
Enregistre les métriques MAE et R2 (calculées sur les données non mises à l'échelle).
Utilisé dans le TP3.

**Entrées (`__init__`) :**
- `modele_pytorch` (nn.Module) : Le modèle de réseau de neurones PyTorch.
- `fonction_perte` (nn.Module) : La fonction de perte (ex: `nn.MSELoss()`).
- `scaler_y` (MinMaxScaler) : Scaler pour la variable cible, utilisé pour dénormaliser les métriques.
- `taux_apprentissage` (float, optionnel) : Taux d'apprentissage pour l'optimiseur. Par défaut 6e-3.
- `classe_optimiseur` (torch.optim.Optimizer, optionnel) : La classe d'optimiseur à utiliser. Par défaut `torch.optim.Adam`.

**Sorties :**
- (Implicitement, pendant l'entraînement/test) Enregistre les métriques dans le logger.

```python
class ModeleLightningBaseRegression(pl.LightningModule):
    """
    Module PyTorch Lightning de base pour les tâches de régression.
    Enregistre les métriques MAE et R2, en utilisant le scaler_y fourni pour les rapporter à l'échelle originale.
    """
    def __init__(self, modele_pytorch: nn.Module, fonction_perte: nn.Module, scaler_y: MinMaxScaler, taux_apprentissage: float = 6e-3, classe_optimiseur = torch.optim.Adam):
        super().__init__()
        self.modele_pytorch = modele_pytorch # Renommé pour éviter conflit avec self.model de PTL
        self.fonction_perte = fonction_perte
        self.scaler_y = scaler_y # Pour dénormaliser les métriques
        self.lr = taux_apprentissage
        self.classe_optimiseur = classe_optimiseur
        self.save_hyperparameters(ignore=['modele_pytorch', 'fonction_perte', 'scaler_y'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modele_pytorch(x)

    def _etape_partagee(self, batch, batch_idx, phase: str):
        x, y_vrai_echelle = batch
        y_pred_echelle = self(x)
        perte = self.fonction_perte(y_pred_echelle, y_vrai_echelle)
        
        # Calculer les métriques sur les données non mises à l'échelle
        metriques_original_scale = calculer_metriques_regression_pytorch(y_vrai_echelle, y_pred_echelle, self.scaler_y)
        
        dict_logs = {
            f"{phase}_perte": perte,
            f"{phase}_mae_original": metriques_original_scale["mae"],
            f"{phase}_r2_original": metriques_original_scale["r2"]
        }
        # prog_bar=True pour les afficher dans la barre de progression
        self.log_dict(dict_logs, on_step=(phase=="train"), on_epoch=True, prog_bar=True, logger=True)
        return perte

    def training_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Le batch peut être juste x, ou (x, y)
        if isinstance(batch, tuple) or isinstance(batch, list):
            x, _ = batch # si y est présent, l'ignorer pour la prédiction
        else:
            x = batch
        y_pred_echelle = self(x)
        # Transformer inversement les prédictions à l'échelle originale
        y_pred_original_np = self.scaler_y.inverse_transform(y_pred_echelle.detach().cpu().numpy())
        return torch.tensor(y_pred_original_np, dtype=torch.float32) # Retourner comme tenseur

    def configure_optimizers(self):
        # Utiliser self.modele_pytorch.parameters() car les paramètres sont dans ce sous-module
        optimiseur = self.classe_optimiseur(self.modele_pytorch.parameters(), lr=self.lr)
        return optimiseur
```

## class ModeleMLP(nn.Module)
Modèle générique de Perceptron Multi-Couches (MLP).
Peut créer des MLP avec un nombre variable de couches, d'unités cachées, de fonctions d'activation et de dropout.
Utilisé pour le TP3 (prédiction de logements) et adaptable pour le TP4 (MNIST) si `dim_entree`/`dim_sortie` sont définis.

**Entrées (`__init__`) :**
- `dim_entree` (int) : Dimensionalité de la couche d'entrée.
- `dim_sortie` (int) : Dimensionalité de la couche de sortie.
- `couches_cachees_dims` (list[int], optionnel) : Liste où chaque entier est le nombre de neurones dans une couche cachée.
  Par défaut à `[64, 32]` (deux couches cachées).
- `fonction_activation` (nn.Module, optionnel) : Fonction d'activation à utiliser après chaque couche cachée.
  Par défaut `nn.ReLU()`. Si `None`, aucune activation n'est appliquée (couches linéaires).
- `taux_dropout` (list[float], optionnel) : Liste des taux de dropout à appliquer après l'activation de chaque couche cachée.
  La longueur doit correspondre à `couches_cachees_dims`. Si `None` ou vide, pas de dropout. Par défaut `None`.
- `fonction_activation_finale` (nn.Module, optionnel) : Fonction d'activation pour la couche de sortie. Par défaut `None`.

**Entrées (`forward`) :**
- `x` (torch.Tensor) : Le tenseur d'entrée. Si le modèle attend une entrée aplatie (ex: pour des images),
l'aplatissement doit se produire avant cet appel ou comme première étape ici.

**Sorties (de `forward`) :**
- (torch.Tensor) : Le tenseur de sortie (prédictions).

```python
from collections import OrderedDict

class ModeleMLP(nn.Module):
    """
    Modèle générique de Perceptron Multi-Couches (MLP).
    Permet de spécifier le nombre de couches cachées, d'unités, d'activation et de dropout.
    """
    def __init__(self, dim_entree: int, dim_sortie: int, 
                 couches_cachees_dims: list = [64, 32], 
                 fonction_activation: nn.Module = nn.ReLU(),
                 taux_dropout: list = None, # Liste de floats, ex: [0.2, 0.1]
                 fonction_activation_finale: nn.Module = None,
                 aplatir_entree: bool = False): # Pour les images par exemple
        super(ModeleMLP, self).__init__()
        
        self.aplatir_entree = aplatir_entree
        couches = OrderedDict()
        dim_actuelle = dim_entree

        for i, h_dim in enumerate(couches_cachees_dims):
            couches[f'fc{i+1}'] = nn.Linear(dim_actuelle, h_dim)
            if fonction_activation is not None:
                couches[f'act{i+1}'] = fonction_activation
            if taux_dropout and i < len(taux_dropout) and taux_dropout[i] > 0:
                couches[f'dropout{i+1}'] = nn.Dropout(taux_dropout[i])
            dim_actuelle = h_dim
        
        couches['fc_sortie'] = nn.Linear(dim_actuelle, dim_sortie)
        if fonction_activation_finale is not None:
            couches['act_finale'] = fonction_activation_finale
            
        self.reseau = nn.Sequential(couches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si x est un batch d'images (ex: [batch_size, C, H, W]), l'aplatir.
        # Ce MLP attend des caractéristiques 1D par échantillon.
        if self.aplatir_entree and x.ndim > 2: 
            x = x.view(x.size(0), -1) 
        return self.reseau(x)

# Exemples d'utilisation pour les modèles du TP3 :
# Pour dim_entree=19 caractéristiques (après prétraitement dans le TP3) et dim_sortie=1 (prix)

# Modèle à une seule couche (linéaire) du TP3
# mlp_1_couche_tp3 = ModeleMLP(dim_entree=19, dim_sortie=1, couches_cachees_dims=[], fonction_activation=None)

# Modèle à 2 couches du TP3 (1 couche cachée de 64 neurones + ReLU, puis couche de sortie)
# mlp_2_couches_tp3 = ModeleMLP(dim_entree=19, dim_sortie=1, couches_cachees_dims=[64], fonction_activation=nn.ReLU())

# Modèle à 4 couches du TP3 (3 couches cachées [128, 64, 32] + ReLU, puis couche de sortie)
# mlp_4_couches_tp3 = ModeleMLP(dim_entree=19, dim_sortie=1, couches_cachees_dims=[128, 64, 32], fonction_activation=nn.ReLU())

# Modèle à 5 couches avec Dropout du TP3
# (4 couches cachées [256, 128, 64, 32] + ReLU, dropout après les 2 premières activations, puis couche de sortie)
# mlp_5_couches_dropout_tp3 = ModeleMLP(dim_entree=19, dim_sortie=1, 
#                                      couches_cachees_dims=[256, 128, 64, 32], 
#                                      fonction_activation=nn.ReLU(),
#                                      taux_dropout=[0.2, 0.2, 0.0, 0.0]) # Dropout après les 2 premières activations ReLU
```

## def visualiser_predictions_regression_pytorch(liste_predictions_batch: list, y_vrai_serie_original: pd.Series, scaler_y: MinMaxScaler, nb_echantillons_a_tracer: int = 100, titre: str = "Prédictions vs Valeurs Réelles (PyTorch)", label_x: str = "Index Maison", label_y: str = "Valeur Maison")
Trace les valeurs vraies vs. prédites pour un modèle de régression PyTorch.
`liste_predictions_batch` est une liste de lots de tenseurs prédits (sortie de `trainer.predict`).
Ces prédictions sont concaténées, converties en NumPy, et transformées inversement en utilisant `scaler_y`.
`y_vrai_serie_original` contient les valeurs vraies originales, non mises à l'échelle.
Utilisé dans le TP3.

**Entrées :**
- `liste_predictions_batch` (list) : Liste de tenseurs PyTorch, où chaque tenseur est un lot de prédictions mises à l'échelle.
- `y_vrai_serie_original` (pd.Series) : Série Pandas des valeurs cibles vraies dans leur échelle originale (non mise à l'échelle).
- `scaler_y` (MinMaxScaler) : L'objet scaler utilisé pour la variable cible, pour la transformation inverse des prédictions.
- `nb_echantillons_a_tracer` (int, optionnel) : Nombre d'échantillons initiaux à tracer. Par défaut 100.
- `titre` (str, optionnel) : Titre du graphique.
- `label_x` (str, optionnel) : Étiquette pour l'axe des x.
- `label_y` (str, optionnel) : Étiquette pour l'axe des y.

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_predictions_regression_pytorch(liste_predictions_batch: list, y_vrai_serie_original: pd.Series, scaler_y: MinMaxScaler, nb_echantillons_a_tracer: int = 100, titre: str = "Prédictions vs Valeurs Réelles (PyTorch)", label_x: str = "Index Maison", label_y: str = "Valeur Maison"):
    """
    Trace les valeurs vraies vs. prédites pour les modèles de régression PyTorch.
    `liste_predictions_batch` contient des lots de prédictions MISES À L'ÉCHELLE issues de trainer.predict().
    `y_vrai_serie_original` contient les valeurs vraies ORIGINALES (non mises à l'échelle).
    Les prédictions sont transformées inversement en utilisant scaler_y.
    """
    # Concaténer tous les tenseurs de prédiction de la liste de lots
    if not liste_predictions_batch:
        print("Avertissement : liste_predictions_batch est vide. Rien à tracer.")
        return
        
    tenseur_toutes_preds_echelle = torch.cat(liste_predictions_batch, dim=0)
    
    # Convertir en NumPy et s'assurer qu'il est 2D pour le scaler
    np_toutes_preds_echelle = tenseur_toutes_preds_echelle.detach().cpu().numpy()
    if np_toutes_preds_echelle.ndim == 1:
        np_toutes_preds_echelle = np_toutes_preds_echelle.reshape(-1, 1)

    # Transformer inversement les prédictions
    y_pred_original_scale = scaler_y.inverse_transform(np_toutes_preds_echelle).flatten()

    # Obtenir les valeurs vraies pour le traçage
    y_vrai_pour_graph = y_vrai_serie_original.values[:nb_echantillons_a_tracer]
    y_pred_pour_graph = y_pred_original_scale[:nb_echantillons_a_tracer]
    
    # S'assurer que les longueurs correspondent pour la création du DataFrame
    nb_echantillons_reel = min(len(y_vrai_pour_graph), len(y_pred_pour_graph))

    df_resultat = pd.DataFrame({
        'Index': list(range(nb_echantillons_reel)),
        'Valeurs Réelles': y_vrai_pour_graph[:nb_echantillons_reel],
        'Prédictions': y_pred_pour_graph[:nb_echantillons_reel]
    })

    plt.figure(figsize=(15, 7))
    sns.lineplot(data=df_resultat, x='Index', y='Valeurs Réelles', marker='o', label='Valeurs Réelles', linestyle='-')
    sns.lineplot(data=df_resultat, x='Index', y='Prédictions', marker='x', label='Prédictions', linestyle='--')

    plt.title(titre)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.show()
```

## class ModeleLightningMNIST(pl.LightningModule)
Module PyTorch Lightning pour la classification MNIST.
Gère les étapes d'entraînement, de validation et de test, en enregistrant la perte et la précision.
Utilisé dans le TP4.

**Entrées (`__init__`) :**
- `modele_pytorch` (nn.Module) : Le réseau de neurones PyTorch (ex: un MLP ou un CNN).
- `classe_optimiseur` (torch.optim.Optimizer) : La classe d'optimiseur (ex: `torch.optim.SGD`, `torch.optim.Adam`).
- `nb_classes` (int, optionnel) : Nombre de classes de sortie (10 pour MNIST). Par défaut 10.
- `taux_apprentissage` (float, optionnel) : Taux d'apprentissage pour l'optimiseur. Par défaut 1e-4.
- `decroissance_poids` (float, optionnel) : Décroissance de poids (pénalité L2) pour l'optimiseur. Par défaut 1e-4.

**Sorties :**
- (Implicitement) Enregistre les métriques, gère l'entraînement.

```python
class ModeleLightningMNIST(pl.LightningModule):
    """
    Module PyTorch Lightning pour la classification MNIST.
    """
    def __init__(self, modele_pytorch: nn.Module, classe_optimiseur, nb_classes: int = 10, taux_apprentissage: float = 1e-4, decroissance_poids: float = 1e-4):
        super().__init__()
        self.modele_pytorch = modele_pytorch # Renommé pour éviter conflit
        self.nb_classes = nb_classes
        self.classe_optimiseur = classe_optimiseur
        self.lr = taux_apprentissage
        self.decroissance_poids = decroissance_poids
        
        # Utilisation de MulticlassAccuracy
        self.train_acc = MulticlassAccuracy(num_classes=nb_classes)
        self.val_acc = MulticlassAccuracy(num_classes=nb_classes)
        self.test_acc = MulticlassAccuracy(num_classes=nb_classes)
        
        self.save_hyperparameters(ignore=['modele_pytorch']) # PTL enregistre les hyperparamètres

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modele_pytorch(x)

    def _etape_partagee(self, batch, batch_idx, phase: str):
        x, y_vrai = batch
        logits = self(x) # Sortie brute du modèle
        perte = F.cross_entropy(logits, y_vrai) # CrossEntropyLoss attend les logits bruts
        
        metrique_acc = getattr(self, f"{phase}_acc")
        acc = metrique_acc(logits, y_vrai) # Passer directement les logits
        
        self.log_dict({f'{phase}_perte': perte, f"{phase}_acc": acc}, 
                      on_step=(phase=="train"), on_epoch=True, prog_bar=True, logger=True)
        return perte

    def training_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._etape_partagee(batch, batch_idx, "test")
        
    # Les réinitialisations de métriques on_..._epoch_end sont gérées par Torchmetrics avec on_epoch=True

    def configure_optimizers(self):
        # self.parameters() inclut les paramètres de self.modele_pytorch
        optimiseur = self.classe_optimiseur(self.parameters(), lr=self.lr, weight_decay=self.decroissance_poids)
        return optimiseur
```

## class ModeleCNNMNIST(nn.Module)
Modèle générique de Réseau de Neurones Convolutifs (CNN), adaptable pour MNIST.
Permet de définir une séquence de blocs convolutifs (Conv2d, Activation, MaxPool, Dropout)
suivie d'une séquence de couches entièrement connectées.
Utilisé pour le TP4.
Note : La configuration des couches doit être soigneusement calculée pour que les dimensions correspondent, en particulier avant la première couche FC.

**Entrées (`__init__`) :**
- `canaux_entree` (int) : Nombre de canaux dans l'image d'entrée (1 pour MNIST en niveaux de gris).
- `nb_classes` (int) : Nombre de classes de sortie (10 pour MNIST).
- `config_couches_conv` (list[dict]) : Configuration pour les couches convolutives.
  Chaque dict : `{'out_channels': int, 'kernel_size': int ou tuple, 'stride': int ou tuple, 'padding': int ou tuple,
                  'pool_kernel_size': int ou tuple (optionnel), 'dropout_rate': float (optionnel)}`
- `config_couches_fc` (list[dict]) : Configuration pour les couches entièrement connectées.
  Chaque dict : `{'out_features': int, 'dropout_rate': float (optionnel)}`
- `fonction_activation` (nn.Module, optionnel) : Fonction d'activation pour les couches conv et FC. Par défaut `nn.ReLU()`.
- `taille_image_initiale` (tuple, optionnel) : (Hauteur, Largeur) de l'image d'entrée, nécessaire si `config_couches_fc`
ne spécifie pas `in_features` pour la première couche FC. Par défaut (28,28) pour MNIST.

**Entrées (`forward`) :**
- `x` (torch.Tensor) : Tenseur d'entrée de forme (batch_size, canaux_entree, H, W).

**Sorties (de `forward`) :**
- (torch.Tensor) : Tenseur de sortie des logits (batch_size, nb_classes).

```python
class ModeleCNNMNIST(nn.Module):
    def __init__(self, canaux_entree: int, nb_classes: int,
                 config_couches_conv: list, # Liste de dictionnaires de configuration
                 config_couches_fc: list,   # Liste de dictionnaires de configuration
                 fonction_activation: nn.Module = nn.ReLU(),
                 taille_image_initiale: tuple = (28, 28)): # (H, W)
        super(ModeleCNNMNIST, self).__init__()

        modules_conv = OrderedDict()
        canaux_actuels = canaux_entree
        h_actuelle, w_actuelle = taille_image_initiale

        for i, cfg in enumerate(config_couches_conv):
            padding = cfg.get('padding', 0)
            stride = cfg.get('stride', 1)
            kernel_size = cfg['kernel_size']
            
            # S'assurer que kernel_size est un tuple pour le calcul des dimensions
            if isinstance(kernel_size, int): kernel_size_h, kernel_size_w = kernel_size, kernel_size
            else: kernel_size_h, kernel_size_w = kernel_size
            
            if isinstance(padding, int): padding_h, padding_w = padding, padding
            else: padding_h, padding_w = padding

            if isinstance(stride, int): stride_h, stride_w = stride, stride
            else: stride_h, stride_w = stride


            modules_conv[f'conv{i+1}'] = nn.Conv2d(
                in_channels=canaux_actuels,
                out_channels=cfg['out_channels'],
                kernel_size=kernel_size, # Peut être int ou tuple
                stride=stride,       # Peut être int ou tuple
                padding=padding      # Peut être int ou tuple
            )
            # Calcul de la dimension de sortie après Conv2d
            h_actuelle = (h_actuelle + 2 * padding_h - kernel_size_h) // stride_h + 1
            w_actuelle = (w_actuelle + 2 * padding_w - kernel_size_w) // stride_w + 1
            
            if fonction_activation:
                modules_conv[f'act_conv{i+1}'] = fonction_activation
            
            if 'pool_kernel_size' in cfg and cfg['pool_kernel_size']:
                pool_kernel = cfg['pool_kernel_size']
                pool_stride = cfg.get('pool_stride', pool_kernel) # Stride = kernel_size par défaut pour MaxPool
                
                if isinstance(pool_kernel, int): pool_kernel_h, pool_kernel_w = pool_kernel, pool_kernel
                else: pool_kernel_h, pool_kernel_w = pool_kernel
                
                if isinstance(pool_stride, int): pool_stride_h, pool_stride_w = pool_stride, pool_stride
                else: pool_stride_h, pool_stride_w = pool_stride

                modules_conv[f'pool{i+1}'] = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
                # Calcul de la dimension de sortie après MaxPool2d
                h_actuelle = h_actuelle // pool_stride_h # Simplifié, MaxPool peut avoir padding/dilation
                w_actuelle = w_actuelle // pool_stride_w

            if 'dropout_rate' in cfg and cfg['dropout_rate'] > 0:
                # Dropout2d est pour les couches conv (agit sur les canaux entiers)
                modules_conv[f'dropout_conv{i+1}'] = nn.Dropout2d(cfg['dropout_rate']) 
            canaux_actuels = cfg['out_channels']

        self.couches_conv = nn.Sequential(modules_conv)
        
        # Calculer la taille aplatie après les couches de convolution
        # Utiliser un tenseur factice pour trouver la forme de sortie des couches_conv
        with torch.no_grad(): # Pas besoin de gradients ici
            entree_factice = torch.randn(1, canaux_entree, taille_image_initiale[0], taille_image_initiale[1])
            forme_sortie_conv = self.couches_conv(entree_factice).shape
            taille_aplatie = forme_sortie_conv[1] * forme_sortie_conv[2] * forme_sortie_conv[3]
            # Equivalent à : taille_aplatie = canaux_actuels * h_actuelle * w_actuelle

        modules_fc = OrderedDict()
        caracteristiques_actuelles = taille_aplatie
        # Construire les couches FC
        for i, cfg_fc in enumerate(config_couches_fc):
            modules_fc[f'fc{i+1}'] = nn.Linear(caracteristiques_actuelles, cfg_fc['out_features'])
            # Pas d'activation sur la dernière couche FC avant softmax/cross_entropy typiquement
            if fonction_activation and i < len(config_couches_fc) - 1 : 
                 modules_fc[f'act_fc{i+1}'] = fonction_activation

            # Pas de dropout sur la dernière couche FC typiquement
            if 'dropout_rate' in cfg_fc and cfg_fc['dropout_rate'] > 0 and i < len(config_couches_fc) -1: 
                modules_fc[f'dropout_fc{i+1}'] = nn.Dropout(cfg_fc['dropout_rate'])
            caracteristiques_actuelles = cfg_fc['out_features']
        
        # S'assurer que la dernière couche sort nb_classes
        if not config_couches_fc or caracteristiques_actuelles != nb_classes :
             modules_fc[f'fc_sortie_finale'] = nn.Linear(caracteristiques_actuelles, nb_classes)

        self.couches_fc = nn.Sequential(modules_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.couches_conv(x)
        x = torch.flatten(x, 1) # Aplatir toutes les dimensions sauf le batch
        x = self.couches_fc(x)
        return x

# Exemple d'utilisation pour les CNN du TP4 :
# cnn_un_conv_config_tp4 = [
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'pool_kernel_size': 2} 
# ]
# fc_un_conv_config_tp4 = [
#     {'out_features': 128}, # La couche finale vers nb_classes est gérée implicitement
# ]
# cnn1_tp4_test = ModeleCNNMNIST(canaux_entree=1, nb_classes=10, 
#                                config_couches_conv=cnn_un_conv_config_tp4, 
#                                config_couches_fc=fc_un_conv_config_tp4, 
#                                taille_image_initiale=(28,28))
# print(torchinfo_summary(cnn1_tp4_test, input_size=(1, 1, 28, 28))) # BS, C, H, W

# cnn_deux_conv_config_tp4 = [
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'dropout_rate': 0.2},
#     {'out_channels': 32, 'kernel_size': 3, 'padding':0, 'dropout_rate': 0.2, 'pool_kernel_size': 2}
# ]
# fc_deux_conv_config_tp4 = [
#     {'out_features': 128, 'dropout_rate': 0.2},
# ]
# cnn2_tp4_test = ModeleCNNMNIST(canaux_entree=1, nb_classes=10, 
#                                config_couches_conv=cnn_deux_conv_config_tp4, 
#                                config_couches_fc=fc_deux_conv_config_tp4, 
#                                taille_image_initiale=(28,28))
# print(torchinfo_summary(cnn2_tp4_test, input_size=(1, 1, 28, 28)))
```

## def creer_dataloaders_classification_images(chemin_dataset: str, taille_batch: int, ratio_train: float, ratio_val: float, taille_img: int = 224, nb_workers: int = 2, pin_memory: bool = True)
Crée des DataLoaders d'entraînement, de validation et de test pour la classification d'images à partir d'une structure de répertoires
(ex: `chemin_dataset/classe_a/image1.jpg`, `chemin_dataset/classe_b/image2.jpg`).
Applique des transformations standard pour l'entraînement (augmentation) et l'évaluation.
Le jeu de données est divisé en ensembles d'entraînement, de validation et de test.
Utilisé dans les TP5, TP6, TP7.

**Entrées :**
- `chemin_dataset` (str) : Chemin vers le répertoire racine du jeu de données.
- `taille_batch` (int) : Nombre d'échantillons par lot.
- `ratio_train` (float) : Proportion du jeu de données à allouer pour l'entraînement (ex: 0.8 pour 80%).
- `ratio_val` (float) : Proportion du jeu de données à allouer pour la validation (ex: 0.1 pour 10%).
L'ensemble de test sera le reste (1 - `ratio_train` - `ratio_val`).
- `taille_img` (int, optionnel) : Taille à laquelle redimensionner les images (taille_img x taille_img). Par défaut 224.
- `nb_workers` (int, optionnel) : Nombre de sous-processus à utiliser pour le chargement des données. Par défaut 2.
- `pin_memory` (bool, optionnel) : Si `True`, le chargeur de données copiera les Tenseurs dans la mémoire épinglée CUDA avant de les retourner. Par défaut `True`.

**Sorties :**
- `loader_train` (DataLoader) : DataLoader pour l'ensemble d'entraînement.
- `loader_val` (DataLoader) : DataLoader pour l'ensemble de validation.
- `loader_test` (DataLoader) : DataLoader pour l'ensemble de test.
- `noms_classes` (list) : Liste des noms de classes trouvés dans le jeu de données.

```python
class SubsetAvecTransform(Dataset):
    """
    Un Dataset qui encapsule un Subset et applique une transformation à la volée.
    """
    def __init__(self, subset_original, transform=None):
        self.subset_original = subset_original
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_original[index] # x est une image PIL ici
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_original)

def creer_dataloaders_classification_images(
    chemin_dataset: str, 
    taille_batch: int, 
    ratio_train: float, 
    ratio_val: float, 
    taille_img: int = 224,
    nb_workers: int = 2, 
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Crée des DataLoaders d'entraînement, de validation et de test pour la classification d'images.
    Divise le jeu de données en fonction des ratios fournis.
    Applique différentes transformations pour les ensembles d'entraînement (avec augmentation) et d'évaluation.
    """

    # Transformations
    moyenne_imagenet = [0.485, 0.456, 0.406]
    std_imagenet = [0.229, 0.224, 0.225]

    transform_train = tv_transforms_v2.Compose([
        tv_transforms_v2.RandomResizedCrop(taille_img, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        tv_transforms_v2.RandomHorizontalFlip(p=0.5),
        tv_transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tv_transforms_v2.RandomRotation(degrees=15),
        tv_transforms_v2.PILToTensor(), 
        tv_transforms_v2.ToDtype(torch.float32, scale=True), 
        tv_transforms_v2.Normalize(mean=moyenne_imagenet, std=std_imagenet)
    ])

    transform_eval = tv_transforms_v2.Compose([
        tv_transforms_v2.Resize(taille_img + 32), 
        tv_transforms_v2.CenterCrop(taille_img),
        tv_transforms_v2.PILToTensor(),
        tv_transforms_v2.ToDtype(torch.float32, scale=True),
        tv_transforms_v2.Normalize(mean=moyenne_imagenet, std=std_imagenet)
    ])

    # Charger le jeu de données complet (sans transformations initialement pour permettre la division)
    dataset_complet_pil = tv_datasets.ImageFolder(chemin_dataset) # Les images sont chargées comme PIL par ImageFolder
    noms_classes = dataset_complet_pil.classes
    
    taille_dataset = len(dataset_complet_pil)
    indices = list(range(taille_dataset))
    np.random.shuffle(indices) # Mélanger les indices avant la division

    # Calculer les tailles des splits
    taille_train = int(ratio_train * taille_dataset)
    taille_val = int(ratio_val * taille_dataset)
    taille_test = taille_dataset - taille_train - taille_val

    if taille_test < 0:
        raise ValueError("La somme de ratio_train et ratio_val ne peut pas dépasser 1.0")
    if taille_train == 0 or taille_val == 0 or taille_test == 0:
         print(f"Avertissement : Un des ensembles (train/val/test) a une taille de 0. Train: {taille_train}, Val: {taille_val}, Test: {taille_test}")
         print("Veuillez ajuster les ratios ou vérifier la taille du dataset.")
         # Gérer ce cas : par exemple, s'assurer qu'il y a au moins 1 échantillon ou lever une erreur plus stricte.
         # Pour cet exemple, on continue, mais cela peut causer des problèmes plus tard.
         if taille_test == 0 and taille_val > 1 : # prendre 1 de val pour test
            taille_val -=1
            taille_test +=1
         elif taille_test == 0 and taille_train > 1: # prendre 1 de train pour test
            taille_train -=1
            taille_test +=1
         # Idem pour val si val est 0

    indices_train = indices[:taille_train]
    indices_val = indices[taille_train : taille_train + taille_val]
    indices_test = indices[taille_train + taille_val:]
    
    # Créer des sous-ensembles (Subsets) de PyTorch
    subset_train_pil = Subset(dataset_complet_pil, indices_train)
    subset_val_pil = Subset(dataset_complet_pil, indices_val)
    subset_test_pil = Subset(dataset_complet_pil, indices_test)

    # Appliquer les transformations aux sous-ensembles en utilisant la classe wrapper
    dataset_train = SubsetAvecTransform(subset_train_pil, transform=transform_train)
    dataset_val = SubsetAvecTransform(subset_val_pil, transform=transform_eval)
    dataset_test = SubsetAvecTransform(subset_test_pil, transform=transform_eval)
    
    # Créer les DataLoaders
    loader_train = DataLoader(dataset_train, batch_size=taille_batch, shuffle=True, num_workers=nb_workers, pin_memory=pin_memory, drop_last=(len(dataset_train) % taille_batch == 1))
    loader_val = DataLoader(dataset_val, batch_size=taille_batch, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory, drop_last=(len(dataset_val) % taille_batch == 1))
    loader_test = DataLoader(dataset_test, batch_size=taille_batch, shuffle=False, num_workers=nb_workers, pin_memory=pin_memory, drop_last=(len(dataset_test) % taille_batch == 1))

    print(f"DataLoaders créés : Train ({len(dataset_train)} échantillons), Val ({len(dataset_val)} échantillons), Test ({len(dataset_test)} échantillons)")
    return loader_train, loader_val, loader_test, noms_classes
```

## class ModeleLightningTransferLearningImage(pl.LightningModule)
Module PyTorch Lightning pour la classification d'images utilisant l'apprentissage par transfert.
Permet d'utiliser des modèles pré-entraînés de `torchvision.models` (ex: VGG, ResNet, EfficientNet, ViT).
Gèle les couches du backbone et remplace la tête de classification finale.
Implémente les étapes d'entraînement, de validation, de test, et enregistre la perte & la précision.
Inclut le calcul et l'affichage de la matrice de confusion à la fin de la phase de test.
Utilisé dans les TP5, TP6, TP7.

**Entrées (`__init__`) :**
- `nom_modele_tv` (str) : Nom du modèle torchvision à utiliser (ex: "vgg16", "resnet50", "efficientnet_b0", "vit_b_16").
- `nb_classes` (int) : Nombre de classes de sortie pour la nouvelle tâche.
- `noms_classes` (list[str], optionnel) : Liste des noms de classes pour les étiquettes de la matrice de confusion. Requis si `tracer_matrice_confusion_test` est `True`.
- `taux_apprentissage` (float, optionnel) : Taux d'apprentissage. Par défaut 3e-4.
- `classe_optimiseur` (torch.optim.Optimizer, optionnel) : Classe d'optimiseur. Par défaut `optim.Adam`.
- `poids_preentraines` (str ou `torchvision.models.WeightsEnum`, optionnel) : Poids pré-entraînés à utiliser.
  "DEFAULT" utilise les meilleurs disponibles. `None` pour une initialisation aléatoire. Des énumérations de Poids spécifiques peuvent aussi être passées. Par défaut "DEFAULT".
- `epoch_degel_backbone` (int, optionnel) : Époque à laquelle dégeler les couches du backbone. -1 pour les garder gelées. Par défaut -1.
- `tracer_matrice_confusion_test` (bool, optionnel) : S'il faut tracer la matrice de confusion après le test. Par défaut `True`.

**Sorties :**
- (Implicitement) Gère l'entraînement, enregistre les métriques.

```python
class ModeleLightningTransferLearningImage(pl.LightningModule):
    def __init__(self, nom_modele_tv: str, nb_classes: int, noms_classes: list = None,
                 taux_apprentissage: float = 3e-4, classe_optimiseur=optim.Adam,
                 poids_preentraines="DEFAULT", epoch_degel_backbone: int = -1, # -1 signifie ne jamais dégeler
                 tracer_matrice_confusion_test: bool = True):
        super().__init__()
        
        self.noms_classes = noms_classes if noms_classes else [str(i) for i in range(nb_classes)]
        # Enregistrer les hyperparamètres pour la journalisation et les points de contrôle
        self.save_hyperparameters() 

        # Charger le modèle pré-entraîné
        if hasattr(tv_models, nom_modele_tv):
            fonction_modele = getattr(tv_models, nom_modele_tv)
            # Utiliser l'API de poids si disponible (PyTorch 0.13+)
            if isinstance(poids_preentraines, str) and poids_preentraines.upper() == "DEFAULT":
                try: # Essayer d'obtenir les poids par défaut via l'API WeightsEnum
                    nom_enum_poids = f"{nom_modele_tv.replace('_','').upper()}_Weights" # ex: VGG16_Weights
                    if hasattr(tv_models, nom_enum_poids):
                         poids_enum = getattr(tv_models, nom_enum_poids).DEFAULT
                         self.backbone = fonction_modele(weights=poids_enum)
                    else: # Fallback pour les modèles plus anciens ou si l'enum n'est pas trouvé
                         self.backbone = fonction_modele(pretrained=True if poids_preentraines else False)
                except AttributeError: # Si WeightsEnum n'existe pas pour ce modèle ou ancienne version de torchvision
                     self.backbone = fonction_modele(pretrained=True if poids_preentraines else False)
            elif poids_preentraines is None: # Pas de poids pré-entraînés
                self.backbone = fonction_modele(weights=None, num_classes=nb_classes if nom_modele_tv.startswith("vit") else None) # ViT s'attend à num_classes à l'init
                if not nom_modele_tv.startswith("vit"): # Pour les autres, modifier la tête après
                    self._adapter_tete_classification(nom_modele_tv, nb_classes)

            else: # Poids spécifiques (objet WeightsEnum) ou booléen (ancien API)
                self.backbone = fonction_modele(weights=poids_preentraines)
        else:
            raise ValueError(f"Modèle {nom_modele_tv} introuvable dans torchvision.models")

        # Geler les paramètres du backbone initialement si des poids pré-entraînés sont utilisés
        if poids_preentraines:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Adapter la tête de classification seulement si on utilise des poids pré-entraînés
            # et que le modèle n'a pas été initialisé avec le bon nb_classes (cas non-ViT)
            if not (nom_modele_tv.startswith("vit") and poids_preentraines is None):
                self._adapter_tete_classification(nom_modele_tv, nb_classes)


        self.critere = nn.CrossEntropyLoss()
        self.train_accuracy = MulticlassAccuracy(num_classes=nb_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=nb_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=nb_classes)
        if self.hparams.tracer_matrice_confusion_test: # Utiliser hparams
            self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=nb_classes)
        
        self.matrice_conf_pour_metriques = None # Pour stocker la matrice de confusion pour TP7 Q9

    def _adapter_tete_classification(self, nom_modele, nb_classes_cible):
        """Adapte la dernière couche linéaire du modèle pour le nb_classes_cible."""
        if "vgg" in nom_modele or "alexnet" in nom_modele:
            if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) > 0:
                 num_ftrs = self.backbone.classifier[-1].in_features
                 self.backbone.classifier[-1] = nn.Linear(num_ftrs, nb_classes_cible)
            else: raise ValueError(f"Structure de classifieur inattendue pour {nom_modele}")
        elif "resnet" in nom_modele or "resnext" in nom_modele or "wide_resnet" in nom_modele or \
             "shufflenet" in nom_modele or "mobilenet" in nom_modele or "mnasnet" in nom_modele:
            if hasattr(self.backbone, 'fc'):
                num_ftrs = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(num_ftrs, nb_classes_cible)
            else: raise ValueError(f"Structure de classifieur inattendue pour {nom_modele} (fc manquant)")
        elif "efficientnet" in nom_modele or "convnext" in nom_modele: # ConvNeXt aussi a .classifier
            if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) > 0:
                # EfficientNet a un Dropout puis un Linear dans classifier
                if isinstance(self.backbone.classifier[-1], nn.Linear):
                    num_ftrs = self.backbone.classifier[-1].in_features
                    self.backbone.classifier[-1] = nn.Linear(num_ftrs, nb_classes_cible)
                else: # Structure inattendue
                     raise ValueError(f"Dernière couche du classifieur de {nom_modele} n'est pas Linear.")
            else: raise ValueError(f"Structure de classifieur inattendue pour {nom_modele}")
        elif "vit" in nom_modele_tv or "swin" in nom_modele_tv:
            # Vision Transformer, Swin Transformer
            # La tête est souvent appelée 'heads' ou 'head'
            if hasattr(self.backbone, 'heads') and hasattr(self.backbone.heads, 'head') and isinstance(self.backbone.heads.head, nn.Linear):
                num_ftrs = self.backbone.heads.head.in_features
                self.backbone.heads.head = nn.Linear(num_ftrs, nb_classes_cible)
            elif hasattr(self.backbone, 'head') and isinstance(self.backbone.head, nn.Linear):
                num_ftrs = self.backbone.head.in_features
                self.backbone.head = nn.Linear(num_ftrs, nb_classes_cible)
            else:
                raise ValueError(f"Impossible de remplacer automatiquement le classifieur pour {nom_modele_tv}. Veuillez adapter.")
        elif "squeezenet" in nom_modele:
            # SqueezeNet a un Conv2d final dans `classifier[1]`
             if hasattr(self.backbone, 'classifier') and len(self.backbone.classifier) > 1 and isinstance(self.backbone.classifier[1], nn.Conv2d):
                in_channels_final_conv = self.backbone.classifier[1].in_channels
                # Remplacer par un Conv2d avec nb_classes_cible comme out_channels, kernel_size=1
                self.backbone.classifier[1] = nn.Conv2d(in_channels_final_conv, nb_classes_cible, kernel_size=1)
                # Il faut aussi s'assurer que la sortie est aplatie correctement après.
                # Souvent, un AdaptiveAvgPool2d((1, 1)) est ajouté après le classifieur pour obtenir (batch, nb_classes, 1, 1)
                # puis un flatten. Pour SqueezeNet, c'est déjà géré.
             else: raise ValueError(f"Structure de classifieur inattendue pour {nom_modele}")
        # Ajouter des cas pour d'autres familles de modèles si nécessaire
        else:
            print(f"Avertissement : Remplacement de classifieur non implémenté de manière spécifique pour {nom_modele_tv}. Tentative générique.")
            # Tentative générique : trouver la dernière couche linéaire et la remplacer.
            # Cela peut être risqué et nécessiter une inspection manuelle du modèle.
            # Pour l'instant, on lève une erreur si non géré explicitement.
            raise ValueError(f"Remplacement de classifieur non implémenté pour {nom_modele_tv}. Veuillez adapter _adapter_tete_classification.")


    def forward(self, x):
        return self.backbone(x)

    def _etape_partagee(self, batch, phase: str):
        x, y = batch
        logits = self(x)
        perte = self.critere(logits, y)
        
        metrique_acc = getattr(self, f"{phase}_accuracy")
        acc = metrique_acc(logits, y)
        
        self.log(f"{phase}_perte", perte, on_step=(phase=="train"), on_epoch=True, prog_bar=True)
        self.log(f"{phase}_acc", acc, on_step=(phase=="train"), on_epoch=True, prog_bar=True)
        
        if phase == "test" and self.hparams.tracer_matrice_confusion_test:
            self.test_confusion_matrix.update(logits.argmax(dim=1), y) # Passer les prédictions (indices) et les vraies étiquettes
            
        return perte

    def training_step(self, batch, batch_idx):
        return self._etape_partagee(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._etape_partagee(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._etape_partagee(batch, "test")

    def on_train_epoch_start(self):
        # Dégeler le backbone à une certaine époque si spécifié
        if self.hparams.epoch_degel_backbone != -1 and self.current_epoch == self.hparams.epoch_degel_backbone:
            print(f"Dégel du backbone à l'époque {self.current_epoch}")
            for param in self.backbone.parameters():
                param.requires_grad = True
            # Il peut être nécessaire de reconfigurer l'optimiseur ou d'ajuster les taux d'apprentissage
            # PTL le fait automatiquement si on change les paramètres qui require_grad.
            # On peut aussi forcer une reconfiguration :
            # self.trainer.strategy.setup_optimizers(self.trainer)
            
    def on_test_epoch_end(self):
        if self.hparams.tracer_matrice_confusion_test:
            cm = self.test_confusion_matrix.compute().cpu().numpy()
            self.matrice_conf_pour_metriques = cm # Stocker pour TP7 Q9

            fig, ax = plt.subplots(figsize=(max(6, self.hparams.nb_classes * 0.8), max(5, self.hparams.nb_classes * 0.6)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.hparams.noms_classes, yticklabels=self.hparams.noms_classes, ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Vrai')
            ax.set_title('Matrice de Confusion')
            plt.tight_layout()
            
            # Enregistrer dans wandb si le logger est disponible
            if self.logger and hasattr(self.logger.experiment, 'log') and isinstance(self.logger, WandbLogger):
                 self.logger.experiment.log({"test_matrice_confusion": wandb.Image(fig)})
            plt.show()
            self.test_confusion_matrix.reset() # Important pour les prochains tests

    def configure_optimizers(self):
        # Filtrer les paramètres pour n'inclure que ceux qui nécessitent des gradients
        # Cela est important si on dégèle progressivement.
        parametres_a_entrainer = filter(lambda p: p.requires_grad, self.parameters())
        optimiseur = self.hparams.classe_optimiseur(parametres_a_entrainer, lr=self.hparams.taux_apprentissage)
        return optimiseur

    def calculer_metriques_classification_depuis_mc(self):
        """Calcule précision, rappel, F1 par classe à partir de self.matrice_conf_pour_metriques. (Pour TP7 Q9)"""
        if self.matrice_conf_pour_metriques is None:
            print("Matrice de confusion non disponible. Exécutez la phase de test d'abord.")
            return None

        mc = self.matrice_conf_pour_metriques
        nb_classes_mc = mc.shape[0]
        metriques = {}

        # Calculer par classe
        for i in range(nb_classes_mc):
            tp = mc[i, i]
            fp = np.sum(mc[:, i]) - tp
            fn = np.sum(mc[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rappel = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) > 0 else 0.0
            
            nom_classe_mc = self.hparams.noms_classes[i] if self.hparams.noms_classes and i < len(self.hparams.noms_classes) else f"Classe_{i}"
            metriques[nom_classe_mc] = {"precision": precision, "rappel": rappel, "f1_score": f1}
        
        # Moyennes macro
        avg_precision = np.mean([m['precision'] for m in metriques.values()])
        avg_rappel = np.mean([m['rappel'] for m in metriques.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metriques.values()])
        metriques['macro_moyenne'] = {"precision": avg_precision, "rappel": avg_rappel, "f1_score": avg_f1}
        
        # Exactitude (Micro F1)
        total_tp = np.sum(np.diag(mc))
        total_echantillons = np.sum(mc)
        exactitude_micro_f1 = total_tp / total_echantillons if total_echantillons > 0 else 0.0
        metriques['exactitude_micro_f1'] = {"precision": exactitude_micro_f1, "rappel": exactitude_micro_f1, "f1_score": exactitude_micro_f1} # Remplir pour la cohérence du DataFrame

        df_metriques = pd.DataFrame.from_dict(metriques, orient='index')
        print("\nIndicateurs de Classification par Classe:")
        print(df_metriques)
        return df_metriques
```

## def visualiser_images_par_classe_depuis_chemin(chemin_dataset: str, nb_images_a_afficher: int = 5, taille_img_affichage: tuple = (128, 128))
Affiche quelques exemples d'images de chaque classe d'un jeu de données structuré en sous-répertoires.
Utilisé dans les TP5, TP6.

**Entrées :**
- `chemin_dataset` (str) : Chemin vers le répertoire racine du jeu de données (ex: "dataset/railway-construction-50/").
- `nb_images_a_afficher` (int, optionnel) : Nombre maximum d'images à afficher par classe. Par défaut 5.
- `taille_img_affichage` (tuple, optionnel) : Taille (largeur, hauteur) à laquelle redimensionner les images pour l'affichage. Par défaut (128, 128).

**Sorties :**
- Aucune (affiche les graphiques).

```python
def visualiser_images_par_classe_depuis_chemin(chemin_dataset: str, nb_images_a_afficher: int = 5, taille_img_affichage: tuple = (128, 128)):
    """
    Affiche quelques exemples d'images de chaque sous-dossier de classe dans le chemin_dataset.
    """
    if not os.path.isdir(chemin_dataset):
        print(f"Erreur : Chemin du jeu de données '{chemin_dataset}' introuvable ou n'est pas un répertoire.")
        return

    noms_classes = [d for d in os.listdir(chemin_dataset) if os.path.isdir(os.path.join(chemin_dataset, d))]
    if not noms_classes:
        print(f"Aucun sous-répertoire (classe) trouvé dans '{chemin_dataset}'.")
        return

    for nom_classe in noms_classes:
        chemin_dossier_classe = os.path.join(chemin_dataset, nom_classe)
        # Chercher les extensions d'images courantes
        chemins_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]: # Ajout de plus d'extensions
            chemins_images.extend(glob.glob(os.path.join(chemin_dossier_classe, ext)))
        
        if not chemins_images:
            print(f"Aucune image trouvée dans le dossier de classe : {chemin_dossier_classe}")
            continue

        print(f"\nAffichage des images pour la classe : {nom_classe}")
        
        nb_colonnes = min(nb_images_a_afficher, 5) # Max 5 colonnes
        nb_lignes = (min(len(chemins_images), nb_images_a_afficher) + nb_colonnes - 1) // nb_colonnes

        plt.figure(figsize=(nb_colonnes * 3, nb_lignes * 3)) # Ajuster figsize
        
        for i, chemin_img in enumerate(chemins_images):
            if i >= nb_images_a_afficher:
                break
            
            try:
                # Utiliser PIL pour ouvrir, puis convertir en tableau numpy pour cv2 si besoin
                img_pil = Image.open(chemin_img).convert('RGB') # S'assurer que c'est RGB
                img_np = np.array(img_pil)
                
                # Redimensionner pour l'affichage (cv2 est efficace pour cela)
                img_redimensionnee = cv2.resize(img_np, taille_img_affichage)
                
                plt.subplot(nb_lignes, nb_colonnes, i + 1)
                plt.imshow(img_redimensionnee)
                plt.title(os.path.basename(chemin_img)[:20]) # Nom de fichier raccourci
                plt.axis('off')
            except Exception as e:
                print(f"Impossible de charger/afficher l'image {chemin_img}: {e}")
        
        plt.tight_layout()
        plt.show()
```

## def tracer_metriques_entrainement_depuis_csv(repertoire_logs: str, nom_logger_csv: str, version_logger_csv: str = '', nom_fichier_metriques: str = 'metrics.csv')
Trace les courbes de perte/précision d'entraînement et de validation à partir d'un fichier `metrics.csv` généré par `CSVLogger` de PyTorch Lightning.
Utilisé dans les TP5, TP6.

**Entrées :**
- `repertoire_logs` (str) : Le répertoire de base où les logs sont stockés (ex: "logs/").
- `nom_logger_csv` (str) : Le 'nom' donné à `CSVLogger` (ex: "cnn1").
- `version_logger_csv` (str, optionnel) : La 'version' donnée à `CSVLogger`. Par défaut ''.
Si `CSVLogger` utilise des sous-répertoires comme 'version_0', 'version_1', l'inclure ici.
- `nom_fichier_metriques` (str, optionnel) : Nom du fichier CSV des métriques. Par défaut 'metrics.csv'.

**Sorties :**
- Aucune (affiche les graphiques).

```python
def tracer_metriques_entrainement_depuis_csv(repertoire_logs: str, nom_logger_csv: str, version_logger_csv: str = '', nom_fichier_metriques: str = 'metrics.csv'):
    """
    Trace les courbes de perte et de précision d'entraînement/validation à partir du fichier metrics.csv de CSVLogger.
    Suppose des noms de colonnes standard comme 'epoch', 'train_loss_epoch', 'val_loss', 
    'train_acc_epoch', 'val_acc'. S'adapte si les noms diffèrent ('train_loss', 'train_acc').
    """
    chemin_metriques = ""
    # Construire le chemin vers le fichier metrics.csv
    if version_logger_csv: # Si la version est spécifiée (ex: "version_0")
        chemin_metriques = os.path.join(repertoire_logs, nom_logger_csv, version_logger_csv, nom_fichier_metriques)
    else: # Si la version est vide, le logger peut enregistrer directement sous nom_logger_csv
          # Ou PTL peut créer un dossier "version_X" automatiquement.
        chemin_base_logger = os.path.join(repertoire_logs, nom_logger_csv)
        if os.path.exists(os.path.join(chemin_base_logger, nom_fichier_metriques)): # Cas simple
            chemin_metriques = os.path.join(chemin_base_logger, nom_fichier_metriques)
        elif os.path.isdir(chemin_base_logger): # Chercher la dernière version_X
            versions = sorted([d for d in os.listdir(chemin_base_logger) if d.startswith("version_") and os.path.isdir(os.path.join(chemin_base_logger,d))])
            if versions:
                 chemin_metriques = os.path.join(chemin_base_logger, versions[-1], nom_fichier_metriques) # Utiliser la dernière version

    if not chemin_metriques or not os.path.exists(chemin_metriques):
        print(f"Erreur : Fichier de métriques introuvable à '{chemin_metriques}' ou dans les versions auto-détectées.")
        print("Veuillez vérifier repertoire_logs, nom_logger_csv et version_logger_csv.")
        return

    try:
        df = pd.read_csv(chemin_metriques)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV '{chemin_metriques}': {e}")
        return

    # Identifier la colonne d'époque, souvent juste 'epoch'
    col_epoch = 'epoch' if 'epoch' in df.columns else None
    if not col_epoch:
        print("Erreur : Colonne 'epoch' introuvable dans metrics.csv.")
        return
        
    # Graphique de la perte
    # Essayer les noms de colonnes communs pour la perte
    col_train_loss = next((col for col in ['train_loss_epoch', 'train_loss'] if col in df.columns), None)
    col_val_loss = next((col for col in ['val_loss_epoch', 'val_loss'] if col in df.columns), None) # val_loss est plus courant que val_loss_epoch

    plt.figure(figsize=(12, 5)) # Créer une seule figure pour les deux subplots

    if col_train_loss and col_val_loss:
        # Nettoyer les NaN et agréger par époque (prendre la moyenne si plusieurs pas par époque)
        df_train_loss = df[df[col_train_loss].notna()].groupby(col_epoch)[col_train_loss].mean().reset_index()
        df_val_loss = df[df[col_val_loss].notna()].groupby(col_epoch)[col_val_loss].mean().reset_index()

        plt.subplot(1, 2, 1)
        plt.plot(df_train_loss[col_epoch], df_train_loss[col_train_loss], label='Perte Entraînement', marker='o')
        plt.plot(df_val_loss[col_epoch], df_val_loss[col_val_loss], label='Perte Validation', marker='x')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.title('Perte Entraînement & Validation')
        plt.legend()
        plt.grid(True)
    else:
        print("Avertissement : Colonnes de perte introuvables pour le traçage.")

    # Graphique de la précision
    col_train_acc = next((col for col in ['train_acc_epoch', 'train_acc'] if col in df.columns), None)
    col_val_acc = next((col for col in ['val_acc_epoch', 'val_acc'] if col in df.columns), None)

    if col_train_acc and col_val_acc:
        df_train_acc = df[df[col_train_acc].notna()].groupby(col_epoch)[col_train_acc].mean().reset_index()
        df_val_acc = df[df[col_val_acc].notna()].groupby(col_epoch)[col_val_acc].mean().reset_index()

        plt.subplot(1, 2, 2) # Le deuxième subplot
        plt.plot(df_train_acc[col_epoch], df_train_acc[col_train_acc], label='Précision Entraînement', marker='o')
        plt.plot(df_val_acc[col_epoch], df_val_acc[col_val_acc], label='Précision Validation', marker='x')
        plt.xlabel('Époque')
        plt.ylabel('Précision')
        plt.title('Précision Entraînement & Validation')
        plt.legend()
        plt.grid(True)
    else:
        print("Avertissement : Colonnes de précision introuvables pour le traçage.")
        if not (col_train_loss and col_val_loss): # Si aucun graphique n'a été fait
            plt.close() # Fermer la figure vide
            return 

    plt.tight_layout()
    plt.show()
```

## def generer_soumission_kaggle_csv(modele: nn.Module, chemin_dossier_test: str, taille_image: int, nom_fichier_csv_sortie: str = "submission.csv", chaine_device: str = "auto")
Génère un fichier CSV pour une soumission Kaggle en prédisant les étiquettes pour les images d'un dossier de test.
Le modèle est supposé être un modèle PyTorch. Des transformations sont appliquées aux images de test.
Utilisé dans les TP5, TP6.

**Entrées :**
- `modele` (nn.Module) : Le modèle PyTorch entraîné (peut être un `LightningModule` ou un `nn.Module` brut).
- `chemin_dossier_test` (str) : Chemin vers le dossier contenant les images de test. Suppose que les images sont directement dans ce dossier.
- `taille_image` (int) : La taille (hauteur et largeur) à laquelle les images de test seront redimensionnées.
- `nom_fichier_csv_sortie` (str, optionnel) : Nom du fichier CSV de sortie. Par défaut "submission.csv".
- `chaine_device` (str, optionnel) : Device pour exécuter les prédictions ("cpu", "cuda", ou "auto"). Par défaut "auto".

**Sorties :**
- `df_soumission` (pd.DataFrame) : DataFrame contenant les ID d'images et les étiquettes prédites.
- Enregistre `df_soumission` dans `nom_fichier_csv_sortie`.

```python
from tqdm import tqdm # Pour la barre de progression

def generer_soumission_kaggle_csv(modele: nn.Module, chemin_dossier_test: str, taille_image: int, nom_fichier_csv_sortie: str = "submission.csv", chaine_device: str = "auto"):
    """
    Génère un fichier CSV de soumission Kaggle en prédisant les étiquettes pour les images d'un dossier de test.
    """
    if chaine_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(chaine_device)
    
    modele.to(device)
    modele.eval() # Mettre le modèle en mode évaluation

    # Transformations standard pour les images de test (cohérentes avec transform_eval)
    moyenne_imagenet = [0.485, 0.456, 0.406] 
    std_imagenet = [0.229, 0.224, 0.225]  
    
    transform_test = tv_transforms_v2.Compose([
        tv_transforms_v2.Resize(taille_image + 32), 
        tv_transforms_v2.CenterCrop(taille_image),
        tv_transforms_v2.PILToTensor(),
        tv_transforms_v2.ToDtype(torch.float32, scale=True), 
        tv_transforms_v2.Normalize(mean=moyenne_imagenet, std=std_imagenet)
    ])

    predictions_liste = []
    noms_fichiers_liste = []

    # Lister uniquement les fichiers image
    fichiers_images = [f for f in os.listdir(chemin_dossier_test) 
                       if os.path.isfile(os.path.join(chemin_dossier_test, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not fichiers_images:
        print(f"Aucun fichier image trouvé dans {chemin_dossier_test}")
        return pd.DataFrame({'ID': [], 'Label': []})

    for nom_fichier_img in tqdm(fichiers_images, desc="Prédiction des images de test"):
        chemin_img = os.path.join(chemin_dossier_test, nom_fichier_img)
        try:
            image_pil = Image.open(chemin_img).convert('RGB') # S'assurer qu'il y a 3 canaux
            tenseur_image = transform_test(image_pil).unsqueeze(0).to(device) # Ajouter dimension batch et déplacer vers device

            with torch.no_grad(): # Pas besoin de gradients pour l'inférence
                logits_sortie = modele(tenseur_image)
                _, idx_classe_predite = torch.max(logits_sortie, 1) # Obtenir l'index du logit max

            predictions_liste.append(idx_classe_predite.item())
            noms_fichiers_liste.append(nom_fichier_img) # Kaggle veut souvent le nom de fichier comme ID
        except Exception as e:
            print(f"Impossible de traiter l'image {chemin_img}: {e}")
            # Optionnellement, ajouter une prédiction de remplacement ou ignorer
            # predictions_liste.append(-1) # Remplacement pour erreur
            # noms_fichiers_liste.append(nom_fichier_img)

    df_soumission = pd.DataFrame({
        'ID': noms_fichiers_liste,    # Ou 'id' selon les exigences de Kaggle
        'Label': predictions_liste # Ou 'target', 'class', etc.
    })

    df_soumission.to_csv(nom_fichier_csv_sortie, index=False)
    print(f"Fichier de soumission '{nom_fichier_csv_sortie}' créé avec succès avec {len(df_soumission)} prédictions.")
    return df_soumission
```

## def visualiser_patches_et_embeddings_vit(modele_vit: nn.Module, chemin_image: str, taille_image: int = 224)
Visualise le processus d'intégration des patchs d'un modèle Vision Transformer (ViT).
Elle montre :
1. L'image originale.
2. L'image divisée en patchs.
3. La similarité des embeddings positionnels.
Cette fonction est principalement destinée à la compréhension pédagogique du traitement d'entrée d'un ViT.
Adapté du TP6.

**Entrées :**
- `modele_vit` (nn.Module) : Une instance d'un modèle ViT (ex: de `torchvision.models` ou un modèle personnalisé
qui a `conv_proj`, `encoder.pos_embedding`, `class_token`).
- `chemin_image` (str) : Chemin vers le fichier image d'entrée.
- `taille_image` (int, optionnel) : Taille à laquelle redimensionner l'image. Par défaut 224.

**Sorties :**
- Aucune (affiche des graphiques).
- Imprime les formes des tenseurs intermédiaires.

```python
def visualiser_patches_et_embeddings_vit(instance_modele_vit: nn.Module, chemin_image: str, taille_image: int = 224):
    """
    Visualise l'intégration des patchs et la similarité des embeddings positionnels pour un modèle ViT.
    Suppose que instance_modele_vit est un modèle ViT PyTorch (ex: de torchvision ou structure similaire).
    """
    # Vérifier si le modèle a les attributs attendus d'un ViT de torchvision
    conv_proj_attr = getattr(instance_modele_vit, 'conv_proj', None)
    encoder_attr = getattr(instance_modele_vit, 'encoder', None)
    pos_embedding_attr = getattr(encoder_attr, 'pos_embedding', None) if encoder_attr else None
    class_token_attr = getattr(instance_modele_vit, 'class_token', None)

    if not (conv_proj_attr and pos_embedding_attr and class_token_attr):
        print("Avertissement : Le modèle fourni ne semble pas avoir les attributs ViT attendus "
              "('conv_proj', 'encoder.pos_embedding', 'class_token'). La visualisation pourrait échouer ou être inexacte.")
        # return # Ou essayer de continuer avec prudence

    # --- 1. Charger et transformer l'image ---
    try:
        img_pil = Image.open(chemin_image).convert('RGB')
    except FileNotFoundError:
        print(f"Erreur : Image introuvable à {chemin_image}")
        return

    # Transformation pour l'entrée ViT
    transform_vers_tenseur = tv_transforms_v2.Compose([
        tv_transforms_v2.Resize((taille_image, taille_image)), 
        tv_transforms_v2.PILToTensor(),
        tv_transforms_v2.ToDtype(torch.float32, scale=True), # Tenseur float [0,1]
        # Les modèles ViT dans torchvision s'attendent souvent à une normalisation spécifique si pré-entraînés sur ImageNet
        # tv_transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tenseur_transforme = transform_vers_tenseur(img_pil).unsqueeze(0) # Ajouter dimension batch
    
    print(f"Taille Image PIL Originale : {img_pil.size}")
    print(f"Forme Tenseur Image Transformée : {img_tenseur_transforme.shape}")

    # --- 2. Intégration des Patchs ---
    instance_modele_vit.eval() 
    with torch.no_grad():
        if conv_proj_attr:
            # `conv_proj` est typique pour le ViT de PyTorch : Conv2d(3, dim_embed, kernel_size=taille_patch, stride=taille_patch)
            embeddings_patch_conv = conv_proj_attr(img_tenseur_transforme) # Forme : [1, dim_embed, nb_patches_h, nb_patches_w]
            print(f"Embeddings de Patch (après conv_proj) forme : {embeddings_patch_conv.shape}")
            
            # Pour la visualisation, nous avons besoin de la taille du patch. Inférer de conv_proj
            taille_patch_h = conv_proj_attr.kernel_size[0]
            taille_patch_w = conv_proj_attr.kernel_size[1]
            nb_patches_h = taille_image // taille_patch_h
            nb_patches_w = taille_image // taille_patch_w
            print(f"Taille de patch inférée : ({taille_patch_h}x{taille_patch_w}), Nb patches : ({nb_patches_h}x{nb_patches_w})")
        else:
            print("Modèle n'a pas 'conv_proj'. Passage des détails de visualisation de l'intégration des patchs.")
            return

    # --- 3. Visualiser les Patchs sur l'Image Originale ---
    img_affichage_np = np.array(img_pil.resize((taille_image, taille_image))) # Pour l'affichage

    fig_patches = plt.figure(figsize=(8, 8))
    fig_patches.suptitle("Image Divisée en Patchs", fontsize=16)
    for i in range(nb_patches_h * nb_patches_w):
        ligne = i // nb_patches_w
        col = i % nb_patches_w
        
        patch = img_affichage_np[ligne*taille_patch_h:(ligne+1)*taille_patch_h, col*taille_patch_w:(col+1)*taille_patch_w]
        
        ax = fig_patches.add_subplot(nb_patches_h, nb_patches_w, i + 1)
        ax.imshow(patch)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

    # --- 4. Embeddings Positionnels ---
    if pos_embedding_attr is not None:
        embed_pos = pos_embedding_attr # Forme : [1, nb_patches + 1, dim_embed] (+1 pour class token)
        print(f"Embeddings Positionnels forme : {embed_pos.shape}")

        # Visualiser la similarité des embeddings positionnels (excluant l'embedding du class token)
        if nb_patches_h * nb_patches_w == embed_pos.shape[1] - 1 : # Vérifier si la grille correspond aux dimensions de l'embed_pos
            embed_pos_patches = embed_pos[0, 1:, :] # Exclure class token
            
            fig_pos_sim = plt.figure(figsize=(min(12, nb_patches_w*0.8), min(12, nb_patches_h*0.8))) # Taille dynamique
            fig_pos_sim.suptitle("Similarités des Embeddings Positionnels", fontsize=16)
            
            # Tracer la similarité cosinus de l'embed_pos de chaque patch avec tous les autres embed_pos de patchs
            for i in range(nb_patches_h * nb_patches_w):
                matrice_sim = F.cosine_similarity(embed_pos_patches[i:i+1, :], embed_pos_patches, dim=1)
                matrice_sim_redim = matrice_sim.reshape((nb_patches_h, nb_patches_w)).detach().cpu().numpy()
                
                ax = fig_pos_sim.add_subplot(nb_patches_h, nb_patches_w, i + 1)
                ax.imshow(matrice_sim_redim, cmap='viridis')
                # ax.set_title(f"Patch {i}", fontsize=8) # Peut rendre le graphique chargé
                ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("Le nombre de patches de la division d'image ne correspond pas aux dimensions des embeddings positionnels. Passage du graphique de similarité.")
    else:
        print("Modèle n'a pas 'encoder.pos_embedding'. Passage de la visualisation des embeddings positionnels.")

    # --- 5. Combiner les Patchs avec les Embeddings Positionnels (Conceptuel du TP) ---
    if conv_proj_attr and class_token_attr and pos_embedding_attr:
        # Les patchs sont [1, dim_embed, nb_patches_h, nb_patches_w]
        # Aplatir et permuter en [1, nb_patches_h*nb_patches_w, dim_embed]
        patches_aplatis = embeddings_patch_conv.flatten(2).transpose(1, 2) 
        
        token_cls = class_token_attr # Forme : [1, 1, dim_embed]
        
        # Concaténer le class token avec les embeddings de patchs
        entree_transformer_sans_pos = torch.cat((token_cls, patches_aplatis), dim=1) # Forme [1, nb_patches+1, dim_embed]
        print(f"Entrée Transformer (token CLS + patchs aplatis) forme : {entree_transformer_sans_pos.shape}")
        
        # Ajouter les embeddings positionnels
        # embed_pos est [1, nb_patches+1, dim_embed]
        if entree_transformer_sans_pos.shape == embed_pos.shape:
            entree_transformer_avec_pos = entree_transformer_sans_pos + embed_pos
            print(f"Entrée Transformer (avec embeddings positionnels) forme : {entree_transformer_avec_pos.shape}")
        else:
            print(f"Incohérence de forme : Entrée sans embed_pos {entree_transformer_sans_pos.shape} vs embed_pos {embed_pos.shape}")

# Exemple (nécessite un modèle ViT et une image) :
# modele_vit_b16 = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
# chemin_image_factice = "chemin_vers_votre_image_exemple.jpg"
# try:
#    Image.new('RGB', (224, 224)).save(chemin_image_factice) # Une image blanche correcte
#    visualiser_patches_et_embeddings_vit(modele_vit_b16, chemin_image_factice)
# except Exception as e:
#    print(f"Erreur dans l'exemple de visualisation ViT : {e}")
```

## def benchmarker_modele_pytorch(modele: nn.Module, forme_entree_exemple: tuple = (1, 3, 224, 224), gpu_uniquement: bool = True, chaine_device: str = "auto")
Évalue un modèle PyTorch pour diverses métriques de performance telles que le temps d'inférence, l'utilisation de la mémoire, les FLOPs, etc.
Utilise la librairie `pytorch_bench` (supposée installée).
Extrait du TP6.

**Entrées :**
- `modele` (nn.Module) : Le modèle PyTorch à évaluer.
- `forme_entree_exemple` (tuple, optionnel) : Forme d'un tenseur d'entrée exemple (Batch, Canaux, Hauteur, Largeur).
  Par défaut (1, 3, 224, 224).
- `gpu_uniquement` (bool, optionnel) : Si `True`, exécute le benchmark uniquement sur GPU (si disponible). Passé à `pytorch_bench.benchmark`.
  Par défaut `True`.
- `chaine_device` (str, optionnel) : Device sur lequel créer l'entrée exemple ("cpu", "cuda", "auto"). Par défaut "auto".

**Sorties :**
- `resultats` (dict ou type pertinent de `pytorch_bench`) : Résultats du benchmark.
- Affiche les résultats.

```python
# Assurez-vous que pytorch_bench est installé : pip install pytorch-bench
# Si l'import échoue, la fonction affichera un message d'erreur.

def benchmarker_modele_pytorch(modele: nn.Module, forme_entree_exemple: tuple = (1, 3, 224, 224), gpu_uniquement: bool = True, chaine_device: str = "auto"):
    """
    Évalue un modèle PyTorch en utilisant la librairie pytorch_bench.
    Assurez-vous que pytorch_bench est installé : pip install pytorch-bench
    """
    try:
        from pytorch_bench import benchmark as executer_benchmark
    except ImportError:
        print("Erreur : Librairie pytorch_bench introuvable. Veuillez l'installer : pip install pytorch-bench")
        print("Passage du benchmark.")
        return None

    if chaine_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(chaine_device)

    if gpu_uniquement and device.type == 'cpu':
        print("GPU non disponible, mais gpu_uniquement=True. Passage du benchmark ou exécution sur CPU si pytorch_bench le gère.")
        # return None # Décommenter pour sauter explicitement si GPU non dispo et requis

    modele.to(device)
    modele.eval() # S'assurer que le modèle est en mode évaluation

    # Créer un tenseur d'entrée exemple sur le bon device
    entree_exemple = torch.randn(forme_entree_exemple).to(device)

    print(f"Benchmarking du modèle {modele.__class__.__name__} sur {device.type.upper()} avec forme d'entrée {forme_entree_exemple}...")
    
    try:
        # La fonction `benchmark` de `pytorch_bench` peut avoir des arguments spécifiques.
        # Le TP montre `gpu_only=True`. Vérifiez la doc de la librairie pour la signature exacte si des problèmes surviennent.
        resultats = executer_benchmark(modele, entree_exemple, gpu_only=(gpu_uniquement and device.type=='cuda'))
        print("\nRésultats du Benchmark:")
        # Le format de `resultats` dépend de pytorch_bench ; typiquement un dict ou un objet personnalisé
        if isinstance(resultats, dict):
            for cle, valeur in resultats.items():
                print(f"  {cle}: {valeur}")
        else:
            print(resultats)
        return resultats
    except Exception as e:
        print(f"Erreur pendant le benchmarking : {e}")
        return None

# Exemple (commenté, nécessite un modèle) :
# modele_cnn_test = tv_models.resnet18() 
# resultats_benchmark = benchmarker_modele_pytorch(modele_cnn_test)
```

## def extraire_images_test_par_classe_depuis_dataloader(loader_test: DataLoader, nb_images_par_classe: int = 5) -> dict
Extrait un nombre spécifié d'images de test (et leurs étiquettes/chemins si disponibles depuis le dataset)
pour chaque classe à partir d'un `DataLoader` PyTorch.
Utile pour l'analyse qualitative ou l'XAI.
Extrait du TP7.

**Entrées :**
- `loader_test` (DataLoader) : `DataLoader` pour l'ensemble de test. Le dataset sous-jacent doit être
un `ImageFolder` ou un `Subset` d'un `ImageFolder`.
- `nb_images_par_classe` (int, optionnel) : Nombre d'images à extraire par classe. Par défaut 5.

**Sorties :**
- `images_par_classe_nom` (dict) : Un dictionnaire où les clés sont les noms de classes et les valeurs sont des listes de tuples.
Chaque tuple : (tenseur_image, idx_etiquette_vraie, chemin_image_si_disponible).

```python
from collections import defaultdict

def extraire_images_test_par_classe_depuis_dataloader(loader_test: DataLoader, nb_images_par_classe: int = 5) -> dict:
    """
    Extrait et organise les images de test par classe à partir d'un DataLoader.
    Tente d'obtenir les chemins des images si le dataset est un ImageFolder.
    """
    images_par_classe_idx = defaultdict(list) # Stocke par index de classe d'abord
    
    # Déterminer les noms de classes et le mapping class_to_idx
    dataset_original = None
    indices_subset = None
    est_subset_imagefolder = False

    if hasattr(loader_test.dataset, 'dataset') and isinstance(loader_test.dataset.dataset, tv_datasets.ImageFolder):
        dataset_original = loader_test.dataset.dataset
        indices_subset = loader_test.dataset.indices # Indices du subset dans le dataset original
        est_subset_imagefolder = True
    elif isinstance(loader_test.dataset, tv_datasets.ImageFolder):
        dataset_original = loader_test.dataset
        indices_subset = list(range(len(dataset_original))) # Tous les indices
    else:
        print("Avertissement : Type de dataset non ImageFolder ou Subset d'ImageFolder. Noms de classes/chemins non récupérables.")
        # On peut quand même essayer de collecter par index d'étiquette si besoin, mais sans noms/chemins.
        # Pour cet exercice, on suppose qu'on a un ImageFolder pour avoir les noms/chemins.
        return {} # Retourner vide si la structure n'est pas celle attendue

    noms_classes = dataset_original.classes
    
    # Itérer sur les indices du subset pour accéder aux images et chemins originaux
    compteurs_collectes = defaultdict(int)
    for idx_original_dataset in indices_subset:
        # Obtenir l'image transformée et l'étiquette
        # dataset_original[idx] applique les transformations définies lors de la création du DataLoader/Dataset
        # Mais ici, loader_test.dataset est déjà le Dataset transformé (SubsetAvecTransform)
        # Donc, pour obtenir l'image transformée comme elle serait dans le batch :
        # On doit trouver l'index correspondant dans loader_test.dataset (qui est le Subset transformé)
        # C'est complexe. Il est plus simple d'itérer sur le DataLoader directement si on n'a pas besoin des chemins.
        # Le code du TP7 pour `get_test_images_by_class` itère sur les indices du `Subset` pour récupérer les chemins.
        # Refaisons cela.
        
        # Si loader_test.dataset est notre SubsetAvecTransform:
        #   loader_test.dataset.subset_original est le Subset de PIL Images
        #   loader_test.dataset.subset_original.dataset est ImageFolder (PIL)
        #   loader_test.dataset.subset_original.indices sont les indices dans ImageFolder
        
        # Pour être cohérent avec le TP7 et obtenir les chemins :
        # On suppose que loader_test.dataset est un objet qui a un attribut .dataset (ImageFolder) et .indices (pour Subset)
        # Ce qui est le cas pour `Subset(ImageFolder(...))`

        # Cette approche est plus fiable pour les chemins :
        if hasattr(loader_test.dataset, 'dataset') and hasattr(loader_test.dataset, 'indices'): # Pour Subset(ImageFolder)
            base_ds = loader_test.dataset.dataset # L'ImageFolder original (images PIL)
            subset_indices_in_base_ds = loader_test.dataset.indices
            
            # On doit trouver quelles images du DataLoader correspondent à quels indices originaux
            # C'est plus simple de juste prendre les premières images du DataLoader et d'essayer de retrouver le chemin si possible
            # Le code du TP7 itère sur les indices du subset et récupère l'image transformée ET le chemin.
            # C'est ce que nous allons faire.
            
            # Réinitialiser la collecte pour cette approche plus précise
            images_par_classe_idx.clear()
            compteurs_collectes.clear()

            for idx_dans_subset_original in range(len(subset_indices_in_base_ds)):
                idx_global_imagefolder = subset_indices_in_base_ds[idx_dans_subset_original]
                
                # Obtenir l'image PIL et l'étiquette de l'ImageFolder
                img_pil, etiquette_idx = base_ds[idx_global_imagefolder] 
                chemin_img = base_ds.samples[idx_global_imagefolder][0]
                
                if compteurs_collectes[etiquette_idx] < nb_images_par_classe:
                    # Appliquer la transformation d'évaluation manuellement ici car on prend l'image PIL
                    # On a besoin de la transform_eval du DataLoader
                    # Supposons que le DataLoader a été créé avec creer_dataloaders_classification_images
                    # qui utilise transform_eval (défini localement dans cette fonction).
                    # Pour une fonction générique, on devrait passer la transform.
                    # Simplification : on va supposer que l'image du dataloader est déjà transformée.
                    # Et on va essayer de prendre les chemins des PREMIERS éléments du dataloader.
                    
                    # Solution plus simple : itérer sur le dataloader, et si c'est un subset d'ImageFolder,
                    # essayer de récupérer le chemin via l'index global.
                    # Mais l'ordre du dataloader n'est pas garanti pour correspondre aux indices du subset de cette manière.
                    
                    # Compromis du TP7 : prendre l'image transformée directement du Dataset
                    # et le chemin via .samples. Cela suppose que loader_test.dataset[i]
                    # correspond à l'image transformée de base_ds.samples[loader_test.dataset.indices[i]].

                    # L'image déjà transformée (issue de loader_test.dataset[idx_subset_transforme])
                    # et son chemin (issue de base_ds.samples[idx_global_imagefolder])
                    # Il faut un moyen de lier idx_subset_transforme à idx_global_imagefolder.
                    # Si loader_test.dataset est SubsetAvecTransform(Subset(ImageFolder)), c'est un peu imbriqué.
                    
                    # L'approche du TP7 :
                    # dataset = test_loader.dataset.dataset  # This is the ImageFolder dataset
                    # subset_indices_tp7 = test_loader.dataset.indices  # These are the indices for our test subset
                    # for idx_tp7 in subset_indices_tp7:
                    #     path, label = dataset.samples[idx_tp7]
                    #     img = dataset[idx_tp7][0] # Get the transformed image

                    # On va émuler cela :
                    img_transformee = loader_test.dataset[idx_dans_subset_original][0] # Appelle __getitem__ de SubsetAvecTransform
                                                                            # qui appelle __getitem__ de Subset
                                                                            # qui appelle __getitem__ de ImageFolder (PIL)
                                                                            # puis applique la transform.
                    
                    images_par_classe_idx[etiquette_idx].append((img_transformee, etiquette_idx, chemin_img))
                    compteurs_collectes[etiquette_idx] += 1

                # Vérifier si toutes les classes ont assez d'images
                if all(c >= nb_images_par_classe for c in compteurs_collectes.values()) and len(compteurs_collectes) == len(noms_classes):
                    break
            break # Sortir de la boucle principale d'itération sur les indices du subset
        else: # Cas où ce n'est pas un Subset d'ImageFolder
            print("Structure de dataset non gérée pour l'extraction de chemins. Chemins seront None.")
            # Fallback : itérer sur le loader et prendre les images sans chemins
            for batch_imgs, batch_labels in loader_test:
                for i in range(len(batch_imgs)):
                    img_t = batch_imgs[i]
                    label_i = batch_labels[i].item()
                    if compteurs_collectes[label_i] < nb_images_par_classe:
                        images_par_classe_idx[label_i].append((img_t, label_i, None))
                        compteurs_collectes[label_i] += 1
                if all(c >= nb_images_par_classe for c in compteurs_collectes.values()) and len(compteurs_collectes) >= len(noms_classes if noms_classes else []): # >= car on ne connaît pas toutes les classes
                    break
            


    # Convertir en dict avec noms de classes comme clés
    images_par_classe_nom = {}
    if noms_classes:
        for etiquette_idx, liste_items in images_par_classe_idx.items():
            if etiquette_idx < len(noms_classes): # S'assurer que l'index est valide
                images_par_classe_nom[noms_classes[etiquette_idx]] = liste_items
            else:
                print(f"Avertissement: Index d'étiquette {etiquette_idx} hors limites pour noms_classes.")
    else: # Utiliser l'index d'étiquette comme clé si les noms de classes ne sont pas connus
        images_par_classe_nom = dict(images_par_classe_idx)
        
    print("\nImages de test extraites par classe:")
    for nom_c, liste_images in images_par_classe_nom.items():
        print(f"  Classe '{nom_c}': {len(liste_images)} images")
            
    return images_par_classe_nom

```

## def expliquer_modele_cnn_captum_torchcam(modele_cnn: nn.Module, tenseur_image_batch: torch.Tensor, idx_etiquette_vraie: int, noms_classes: list, nom_methode_xai: str, chemin_image_originale: str = None, chaine_device: str = "auto")
Explique la prédiction d'un modèle CNN sur une seule image en utilisant des méthodes XAI spécifiées
de Captum (GradientSHAP, Occlusion, IntegratedGradients, DeepLift) ou
TorchCAM (GradCAM, LayerCAM).
Extrait du TP7.

**Entrées :**
- `modele_cnn` (nn.Module) : Le modèle CNN PyTorch entraîné (ex: une instance EfficientNet).
  Suppose qu'il s'agit d'un `ModeleLightningTransferLearningImage` ou qu'il a `modele.features[-1]` pour les méthodes CAM.
- `tenseur_image_batch` (torch.Tensor) : Tenseur image d'entrée, forme attendue (1, C, H, W).
- `idx_etiquette_vraie` (int) : L'index de classe vrai de l'image d'entrée.
- `noms_classes` (list) : Liste des noms de classes, où l'index correspond à l'index de classe.
- `nom_methode_xai` (str) : Nom de la méthode XAI à utiliser.
  Supportées : "gradientshap", "occlusion", "integratedgradients", "deeplift", "gradcam", "layercam".
- `chemin_image_originale` (str, optionnel) : Chemin vers le fichier image original (utilisé pour un des sous-graphiques). Par défaut `None`.
- `chaine_device` (str, optionnel) : Device sur lequel exécuter ("cpu", "cuda", "auto"). Par défaut "auto".

**Sorties :**
- Aucune (affiche un graphique avec l'image originale, la carte d'attribution XAI, la superposition, et la prédiction).

```python
def expliquer_modele_cnn_captum_torchcam(
    modele_cnn: nn.Module, 
    tenseur_image_batch: torch.Tensor, 
    idx_etiquette_vraie: int, 
    noms_classes: list, 
    nom_methode_xai: str, 
    chemin_image_originale: str = None, 
    chaine_device: str = "auto"):
    """
    Explique la prédiction d'un modèle CNN en utilisant des méthodes Captum ou TorchCAM.
    `modele_cnn` peut être un nn.Module brut ou un pl.LightningModule.
    `tenseur_image_batch` doit être un tenseur d'image unique avec dimension batch : (1, C, H, W).
    """

    if chaine_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(chaine_device)
    
    modele_interne_pour_cam = modele_cnn # Modèle à utiliser pour les extracteurs CAM
    if isinstance(modele_cnn, pl.LightningModule):
        if hasattr(modele_cnn, 'modele_pytorch') and isinstance(modele_cnn.modele_pytorch, nn.Module):
            modele_interne_pour_cam = modele_cnn.modele_pytorch
        elif hasattr(modele_cnn, 'backbone') and isinstance(modele_cnn.backbone, nn.Module):
             modele_interne_pour_cam = modele_cnn.backbone
    
    modele_cnn.to(device)
    modele_cnn.eval()
    
    img_entree = tenseur_image_batch.clone().detach().to(device).requires_grad_(True)

    with torch.no_grad():
        logits_sortie = modele_cnn(img_entree) 
    
    probs_sortie = F.softmax(logits_sortie, dim=1)
    score_prediction, tenseur_idx_etiquette_pred = torch.topk(probs_sortie, 1)
    idx_etiquette_pred = tenseur_idx_etiquette_pred.item()

    np_attributions = None 
    tenseur_attr = None # Pour Captum

    # --- Méthodes Captum ---
    if nom_methode_xai.lower() == "gradientshap":
        gs = GradientShap(modele_cnn)
        # Lignes de base : bruit aléatoire, zéros, image floutée, etc.
        # Le TP7 utilisait `rand_img_dist = torch.rand(50, 3, 224, 224).to(device)`
        # Pour la simplicité, utilisons des zéros et des uns comme lignes de base
        lignes_base_gs = torch.cat([torch.zeros_like(img_entree), torch.ones_like(img_entree)], dim=0).to(device)
        tenseur_attr = gs.attribute(img_entree, baselines=lignes_base_gs, target=idx_etiquette_pred, n_samples=10, stdevs=0.001)
    elif nom_methode_xai.lower() == "occlusion":
        occ = Occlusion(modele_cnn)
        # Adapter strides et sliding_window_shapes à la taille de l'image et des caractéristiques
        # Par exemple, pour une image 224x224, des patchs de 16x16
        strides_occ = (3, img_entree.shape[2]//14, img_entree.shape[3]//14) # Ex: (3, 16, 16)
        sws_occ = (3, img_entree.shape[2]//14, img_entree.shape[3]//14) # Ex: (3, 16, 16)
        tenseur_attr = occ.attribute(img_entree, target=idx_etiquette_pred, strides=strides_occ, 
                                      sliding_window_shapes=sws_occ, baselines=0)
    elif nom_methode_xai.lower() == "integratedgradients":
        ig = IntegratedGradients(modele_cnn)
        ligne_base_ig = torch.zeros_like(img_entree).to(device)
        tenseur_attr = ig.attribute(img_entree, baselines=ligne_base_ig, target=idx_etiquette_pred)
    elif nom_methode_xai.lower() == "deeplift":
        dl = DeepLift(modele_cnn)
        ligne_base_dl = torch.zeros_like(img_entree).to(device)
        tenseur_attr = dl.attribute(img_entree, baselines=ligne_base_dl, target=idx_etiquette_pred)
    
    if tenseur_attr is not None: # Si une méthode Captum a été utilisée
         np_attributions = tenseur_attr.sum(dim=1).squeeze(0).cpu().detach().numpy() # Somme sur les canaux

    # --- Méthodes TorchCAM ---
    elif nom_methode_xai.lower() in ["gradcam", "layercam"]:
        couche_cible_cam = None
        # Tenter de trouver une couche cible appropriée (souvent la dernière couche conv)
        # Cette logique est spécifique au modèle et peut nécessiter une adaptation.
        if hasattr(modele_interne_pour_cam, 'features') and isinstance(modele_interne_pour_cam.features, nn.Sequential) and len(modele_interne_pour_cam.features)>0: # Ex: EfficientNet, VGG
            # Trouver la dernière couche Conv2d ou BatchNorm2d avant la piscine ou le classifieur
            for layer_module in reversed(list(modele_interne_pour_cam.features)):
                 if isinstance(layer_module, (nn.Conv2d, nn.BatchNorm2d)): # Prendre le bloc qui contient Conv/BN
                    couche_cible_cam = layer_module
                    break
            if couche_cible_cam is None and len(modele_interne_pour_cam.features) > 1: # Fallback
                 couche_cible_cam = modele_interne_pour_cam.features[-2] if len(modele_interne_pour_cam.features) > 1 else modele_interne_pour_cam.features[-1]

        elif hasattr(modele_interne_pour_cam, 'layer4') and isinstance(modele_interne_pour_cam.layer4, nn.Sequential): # Ex: ResNet
            couche_cible_cam = modele_interne_pour_cam.layer4[-1] # Le dernier bloc de layer4
        
        if not couche_cible_cam:
            print(f"Erreur : Couche cible pour CAM n'a pas pu être trouvée pour {modele_interne_pour_cam.__class__.__name__}. CAM échouera.")
            return

        extracteur_cam = None
        if nom_methode_xai.lower() == "gradcam":
            extracteur_cam = GradCAM(modele_interne_pour_cam, target_layer=couche_cible_cam)
        else: # layercam
            extracteur_cam = LayerCAM(modele_interne_pour_cam, target_layer=couche_cible_cam)
        
        # Les extracteurs CAM de torchcam nécessitent l'index de classe et les scores (logits)
        # On utilise les logits_sortie du modèle principal (modele_cnn)
        liste_map_activation = extracteur_cam(logits_sortie.squeeze(0).argmax().item(), logits_sortie) 
        extracteur_cam.remove_hooks() # Important!
        if liste_map_activation and len(liste_map_activation) > 0:
            np_attributions = liste_map_activation[0].cpu().numpy() # Déjà [H, W]
        else:
            print("Erreur : L'extracteur CAM n'a pas retourné de carte d'activation.")
            return
    else:
        raise ValueError(f"Méthode XAI non supportée : {nom_methode_xai}")

    # --- Traçage ---
    if np_attributions is None:
        print("Carte d'attribution non générée.")
        return
        
    # Normaliser les attributions pour l'affichage
    attributions_norm = (np_attributions - np_attributions.min()) / (np_attributions.max() - np_attributions.min() + 1e-9)

    img_originale_pil = F_to_pil_image(img_entree.squeeze(0).cpu()) 
    map_attribution_pil = F_to_pil_image(attributions_norm, mode='F') # Mode 'F' pour float niveaux de gris

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_originale_pil)
    axes[0].set_title("Image Originale")
    axes[0].axis('off')

    axes[1].imshow(attributions_norm, cmap='jet')
    axes[1].set_title(f"Carte {nom_methode_xai.capitalize()}")
    axes[1].axis('off')

    axes[2].imshow(torchcam_overlay_mask(img_originale_pil, map_attribution_pil, alpha=0.5))
    axes[2].set_title("Superposition")
    axes[2].axis('off')
    
    nom_classe_actuelle = noms_classes[idx_etiquette_vraie]
    nom_classe_predite = noms_classes[idx_etiquette_pred]
    est_correct = (nom_classe_predite == nom_classe_actuelle)

    # Utiliser l'image chargée avec OpenCV pour le texte si le chemin est fourni
    if chemin_image_originale and os.path.exists(chemin_image_originale):
        img_pour_texte_cv2 = cv2.imread(chemin_image_originale)
        img_pour_texte_cv2 = cv2.cvtColor(img_pour_texte_cv2, cv2.COLOR_BGR2RGB)
    else: # Fallback sur le tenseur si le chemin n'est pas bon
        img_pour_texte_cv2 = np.array(img_originale_pil) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Échelle de police dynamique basée sur la taille de l'image
    echelle_font = min(img_pour_texte_cv2.shape[0], img_pour_texte_cv2.shape[1]) / 450 
    epaisseur = max(1, int(echelle_font * 1.8)) # Épaisseur dynamique
    
    texte_pred = f"Pred: {nom_classe_predite} ({score_prediction.item():.2f})"
    texte_vrai = f"Vrai: {nom_classe_actuelle}"

    (tw1, th1), _ = cv2.getTextSize(texte_pred, font, echelle_font, epaisseur)
    (tw2, th2), _ = cv2.getTextSize(texte_vrai, font, echelle_font, epaisseur)

    # Dessiner le texte sur l'image
    y_pos = th1 + 10
    cv2.putText(img_pour_texte_cv2, texte_pred, (10, y_pos), font, echelle_font, (0, 255, 0) if est_correct else (255, 0, 0), epaisseur, cv2.LINE_AA)
    y_pos += th2 + 10
    cv2.putText(img_pour_texte_cv2, texte_vrai, (10, y_pos), font, echelle_font, (255, 255, 255), epaisseur, cv2.LINE_AA) # Vrai en blanc

    axes[3].imshow(img_pour_texte_cv2)
    axes[3].set_title("Prédiction")
    axes[3].axis('off')

    fig.suptitle(f"XAI: {nom_methode_xai.capitalize()} | Correct: {est_correct}", fontsize=16, color='green' if est_correct else 'red')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Ajuster pour le suptitle
    plt.show()

```

## def expliquer_modele_vit_captum_tis(modele_vit: nn.Module, tenseur_image_batch: torch.Tensor, idx_etiquette_vraie: int, noms_classes: list, nom_methode_xai: str, chemin_image_originale: str = None, chaine_device: str = "auto", tis_nb_masques: int = 1024, tis_taille_batch: int = 128)
Explique la prédiction d'un modèle Vision Transformer (ViT) sur une seule image.
Utilise des méthodes XAI spécifiées de Captum (GradientSHAP, Occlusion, IntegratedGradients, DeepLift)
ou la méthode TIS (Transformer Input Sampling).
Extrait du TP7.

**Entrées :**
- `modele_vit` (nn.Module) : Le modèle ViT PyTorch entraîné.
  Suppose qu'il s'agit d'un `ModeleLightningTransferLearningImage` ou d'un ViT brut.
- `tenseur_image_batch` (torch.Tensor) : Tenseur image d'entrée, forme (1, C, H, W).
- `idx_etiquette_vraie` (int) : L'index de classe vrai de l'image d'entrée.
- `noms_classes` (list) : Liste des noms de classes.
- `nom_methode_xai` (str) : Nom de la méthode XAI.
  Supportées : "gradientshap", "occlusion", "integratedgradients", "deeplift", "tis".
- `chemin_image_originale` (str, optionnel) : Chemin vers le fichier image original pour le traçage. Par défaut `None`.
- `chaine_device` (str, optionnel) : Device ("cpu", "cuda", "auto"). Par défaut "auto".
- `tis_nb_masques` (int, optionnel) : Nombre de masques pour TIS. Par défaut 1024.
- `tis_taille_batch` (int, optionnel) : Taille de batch pour TIS. Par défaut 128.

**Sorties :**
- Aucune (affiche un graphique avec l'image originale, l'attribution XAI, la superposition, et la prédiction).

```python
# TIS nécessite le clonage/l'installation de sa librairie spécifique (fait dans les instructions du TP7)
# from Transformer_Input_Sampling.tis import TIS # Supposé importable

def expliquer_modele_vit_captum_tis(
    modele_vit: nn.Module, 
    tenseur_image_batch: torch.Tensor, 
    idx_etiquette_vraie: int, 
    noms_classes: list, 
    nom_methode_xai: str, 
    chemin_image_originale: str = None, 
    chaine_device: str = "auto",
    tis_nb_masques: int = 1024, 
    tis_taille_batch: int = 128):
    """
    Explique la prédiction d'un modèle ViT en utilisant des méthodes Captum ou TIS.
    `modele_vit` peut être un nn.Module brut ou un pl.LightningModule.
    `tenseur_image_batch` doit être un tenseur d'image unique avec dimension batch : (1, C, H, W).
    """
    
    if chaine_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(chaine_device)

    modele_vit.to(device)
    modele_vit.eval()
    
    img_entree = tenseur_image_batch.clone().detach().to(device).requires_grad_(True)

    with torch.no_grad():
        logits_sortie = modele_vit(img_entree)
    probs_sortie = F.softmax(logits_sortie, dim=1)
    score_prediction, tenseur_idx_etiquette_pred = torch.topk(probs_sortie, 1)
    idx_etiquette_pred = tenseur_idx_etiquette_pred.item()

    np_attributions = None
    tenseur_attr = None 

    # --- Méthodes Captum (identiques à celles pour CNN) ---
    if nom_methode_xai.lower() == "gradientshap":
        gs = GradientShap(modele_vit)
        lignes_base_gs = torch.cat([torch.zeros_like(img_entree), torch.ones_like(img_entree)], dim=0).to(device)
        tenseur_attr = gs.attribute(img_entree, baselines=lignes_base_gs, target=idx_etiquette_pred, n_samples=10, stdevs=0.001)
    elif nom_methode_xai.lower() == "occlusion":
        occ = Occlusion(modele_vit)
        # Pour ViT, la taille de la fenêtre glissante devrait idéalement correspondre à la taille d'un patch
        # Supposons une taille de patch de 16x16 pour une image de 224x224 (14x14 patches)
        patch_size = img_entree.shape[2] // 14 # Approximation
        strides_occ = (3, patch_size, patch_size) 
        sws_occ = (3, patch_size, patch_size) 
        tenseur_attr = occ.attribute(img_entree, target=idx_etiquette_pred, strides=strides_occ, 
                                      sliding_window_shapes=sws_occ, baselines=0)
    elif nom_methode_xai.lower() == "integratedgradients":
        ig = IntegratedGradients(modele_vit)
        ligne_base_ig = torch.zeros_like(img_entree).to(device)
        tenseur_attr = ig.attribute(img_entree, baselines=ligne_base_ig, target=idx_etiquette_pred)
    elif nom_methode_xai.lower() == "deeplift":
        dl = DeepLift(modele_vit)
        ligne_base_dl = torch.zeros_like(img_entree).to(device)
        tenseur_attr = dl.attribute(img_entree, baselines=ligne_base_dl, target=idx_etiquette_pred)
    
    if tenseur_attr is not None: 
         np_attributions = tenseur_attr.sum(dim=1).squeeze(0).cpu().detach().numpy()

    # --- Méthode TIS ---
    elif nom_methode_xai.lower() == "tis":
        try:
            from Transformer_Input_Sampling.tis import TIS # Importation locale
            
            modele_interne_vit = modele_vit # Modèle nn.Module brut pour TIS
            if isinstance(modele_vit, pl.LightningModule):
                if hasattr(modele_vit, 'modele_pytorch') and isinstance(modele_vit.modele_pytorch, nn.Module):
                    modele_interne_vit = modele_vit.modele_pytorch
                elif hasattr(modele_vit, 'backbone') and isinstance(modele_vit.backbone, nn.Module):
                     modele_interne_vit = modele_vit.backbone
                else:
                    print("Avertissement : Impossible d'obtenir le nn.Module interne du LightningModule pour TIS. Utilisation du LightningModule lui-même.")
            
            explainer_tis = TIS(modele_interne_vit, n_masks=tis_nb_masques, batch_size=tis_taille_batch, verbose=False)
            # TIS retourne directement la carte d'attribution, forme (H, W) ou (1, H, W)
            map_attr_tis = explainer_tis(img_entree).cpu().detach() # img_entree est déjà sur device
            if map_attr_tis.ndim == 3 and map_attr_tis.shape[0] == 1: # Si [1, H, W]
                np_attributions = map_attr_tis.squeeze(0).numpy()
            elif map_attr_tis.ndim == 2: # Si [H, W]
                 np_attributions = map_attr_tis.numpy()
            else:
                print(f"Forme de sortie TIS inattendue : {map_attr_tis.shape}")
                return
        except ImportError:
            print("Erreur : Librairie Transformer_Input_Sampling (TIS) introuvable ou échec de l'import.")
            print("Veuillez vous assurer qu'elle est installée et accessible (ex: clonée depuis GitHub comme par TP7).")
            return
        except Exception as e:
            print(f"Erreur pendant l'explication TIS : {e}")
            return
    else:
        raise ValueError(f"Méthode XAI non supportée pour ViT : {nom_methode_xai}")

    # --- Traçage (identique à celui pour CNN) ---
    if np_attributions is None:
        print("Carte d'attribution non générée.")
        return
        
    attributions_norm = (np_attributions - np_attributions.min()) / (np_attributions.max() - np_attributions.min() + 1e-9)

    img_originale_pil = F_to_pil_image(img_entree.squeeze(0).cpu())
    map_attribution_pil = F_to_pil_image(attributions_norm, mode='F')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_originale_pil); axes[0].set_title("Image Originale"); axes[0].axis('off')
    axes[1].imshow(attributions_norm, cmap='jet'); axes[1].set_title(f"Carte {nom_methode_xai.capitalize()}"); axes[1].axis('off')
    
    try: # Utiliser torchcam_overlay_mask si disponible
        axes[2].imshow(torchcam_overlay_mask(img_originale_pil, map_attribution_pil, alpha=0.5))
    except NameError: # Fallback si torchcam_overlay_mask n'est pas défini (ou importé comme tel)
        print("torchcam_overlay_mask non disponible, affichage d'un simple mélange de carte de chaleur.")
        melange = np.array(img_originale_pil, dtype=float) * 0.5
        couleur_heatmap = (plt.cm.jet(attributions_norm)[:, :, :3] * 255).astype(float)
        melange += couleur_heatmap * 0.5
        axes[2].imshow(np.clip(melange, 0, 255).astype(np.uint8))

    axes[2].set_title("Superposition"); axes[2].axis('off')
    
    nom_classe_actuelle = noms_classes[idx_etiquette_vraie]
    nom_classe_predite = noms_classes[idx_etiquette_pred]
    est_correct = (nom_classe_predite == nom_classe_actuelle)

    if chemin_image_originale and os.path.exists(chemin_image_originale):
        img_pour_texte_cv2 = cv2.imread(chemin_image_originale)
        img_pour_texte_cv2 = cv2.cvtColor(img_pour_texte_cv2, cv2.COLOR_BGR2RGB)
    else:
        img_pour_texte_cv2 = np.array(img_originale_pil)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    echelle_font = min(img_pour_texte_cv2.shape[0], img_pour_texte_cv2.shape[1]) / 450
    epaisseur = max(1, int(echelle_font * 1.8))
    texte_pred = f"Pred: {nom_classe_predite} ({score_prediction.item():.2f})"
    texte_vrai = f"Vrai: {nom_classe_actuelle}"
    (tw1, th1), _ = cv2.getTextSize(texte_pred, font, echelle_font, epaisseur)
    (tw2, th2), _ = cv2.getTextSize(texte_vrai, font, echelle_font, epaisseur)
    y_pos = th1 + 10
    cv2.putText(img_pour_texte_cv2, texte_pred, (10, y_pos), font, echelle_font, (0, 255, 0) if est_correct else (255, 0, 0), epaisseur, cv2.LINE_AA)
    y_pos += th2 + 10
    cv2.putText(img_pour_texte_cv2, texte_vrai, (10, y_pos), font, echelle_font, (255,255,255), epaisseur, cv2.LINE_AA)

    axes[3].imshow(img_pour_texte_cv2); axes[3].set_title("Prédiction"); axes[3].axis('off')
    fig.suptitle(f"XAI (ViT): {nom_methode_xai.capitalize()} | Correct: {est_correct}", fontsize=16, color='green' if est_correct else 'red')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()
```

## def creer_masques_rise(nb_masques: int, petit_h_masque: int, petit_l_masque: int, h_image: int, l_image: int, proba_p1: float = 0.5) -> torch.Tensor
Génère un lot de masques binaires aléatoires pour la méthode RISE (Randomized Input Sampling for Explanation).
Chaque petit masque est suréchantillonné à la taille de l'image et recadré aléatoirement.
Extrait du TP7 (Partie Optionnelle).

**Entrées :**
- `nb_masques` (int) : Nombre de masques aléatoires à générer.
- `petit_h_masque` (int) : Hauteur du petit masque aléatoire initial.
- `petit_l_masque` (int) : Largeur du petit masque aléatoire initial.
- `h_image` (int) : Hauteur de l'image cible (et taille finale du masque).
- `l_image` (int) : Largeur de l'image cible (et taille finale du masque).
- `proba_p1` (float, optionnel) : Probabilité d'un '1' dans le petit masque binaire. Par défaut 0.5.

**Sorties :**
- `tenseur_masques` (torch.Tensor) : Un tenseur de forme (nb_masques, h_image, l_image) contenant les masques générés,
normalisés à [0, 1].

```python
def creer_masques_rise(nb_masques: int, petit_h_masque: int, petit_l_masque: int, h_image: int, l_image: int, proba_p1: float = 0.5) -> torch.Tensor:
    """
    Génère des masques binaires aléatoires, les suréchantillonne et les recadre pour RISE.
    """
    liste_masques = []
    for _ in range(nb_masques):
        # 1. Créer un petit masque binaire aléatoire (0s et 1s)
        petit_masque_np = np.random.choice([0, 1], size=(petit_h_masque, petit_l_masque), p=[1 - proba_p1, proba_p1])
        
        # 2. Suréchantillonner le petit masque pour qu'il soit légèrement plus grand que l'image
        # Le TP7 code utilise (h_image + petit_h_masque, l_image + petit_l_masque) pour redimensionner
        
        h_sur_echant = h_image + petit_h_masque 
        l_sur_echant = l_image + petit_l_masque

        # Convertir en Image PIL, puis redimensionner
        petit_masque_pil = Image.fromarray((petit_masque_np * 255).astype(np.uint8), mode='L') # 'L' pour niveaux de gris
        masque_sur_echant_pil = petit_masque_pil.resize((l_sur_echant, h_sur_echant), resample=Image.Resampling.BILINEAR)
        masque_sur_echant_np = np.array(masque_sur_echant_pil, dtype=np.float32) / 255.0 # Retour à [0,1] float

        # 3. Recadrer aléatoirement le masque suréchantillonné à la taille de l'image
        rec_y = np.random.randint(0, petit_h_masque + 1) # y_début max pour le recadrage
        rec_x = np.random.randint(0, petit_l_masque + 1) # x_début max pour le recadrage
        
        masque_final_np = masque_sur_echant_np[rec_y : rec_y + h_image, rec_x : rec_x + l_image]
        
        # Normaliser (devrait déjà être proche de [0,1] après /255.0)
        min_val, max_val = masque_final_np.min(), masque_final_np.max()
        if max_val > min_val: # Éviter la division par zéro si le masque est plat
            masque_final_np = (masque_final_np - min_val) / (max_val - min_val)
        else: # Si plat, c'est probablement tout 0 ou tout 1 à cause de proba_p1 étant 0 ou 1.
            masque_final_np = np.ones_like(masque_final_np) * min_val # Le garder plat

        liste_masques.append(torch.from_numpy(masque_final_np))
        
    return torch.stack(liste_masques) # Forme : (nb_masques, h_image, l_image)

```

## def appliquer_masques_rise_et_predire(modele: nn.Module, tenseur_image_originale: torch.Tensor, tenseur_masques_rise: torch.Tensor, taille_batch_preds: int = 32, chaine_device: str = "auto") -> torch.Tensor
Applique un ensemble de masques RISE à une image originale, créant des images masquées,
puis obtient les prédictions d'un modèle pour ces images masquées.
Extrait du TP7 (Partie Optionnelle).
Modifié pour ne retourner que les prédictions, car les images masquées peuvent être recréées ou ne sont pas toujours nécessaires pour la suite.

**Entrées :**
- `modele` (nn.Module) : Le modèle PyTorch à utiliser pour les prédictions.
- `tenseur_image_originale` (torch.Tensor) : L'image d'entrée originale, forme (C, H, W) ou (1, C, H, W).
- `tenseur_masques_rise` (torch.Tensor) : Tenseur des masques RISE, forme (NbMasques, H, W).
- `taille_batch_preds` (int, optionnel) : Taille de batch pour exécuter les prédictions afin de gérer la mémoire. Par défaut 32.
- `chaine_device` (str, optionnel) : Device ("cpu", "cuda", "auto"). Par défaut "auto".

**Sorties :**
- `tenseur_predictions` (torch.Tensor) : Tenseur des prédictions du modèle (probabilités softmax) pour les images masquées,
forme (NbMasques, NbClasses).

```python
def appliquer_masques_rise_et_predire(
    modele: nn.Module, 
    tenseur_image_originale: torch.Tensor, # Forme (C, H, W) ou (1, C, H, W)
    tenseur_masques_rise: torch.Tensor,     # Forme (NbMasques, H, W)
    taille_batch_preds: int = 32, 
    chaine_device: str = "auto"
) -> torch.Tensor: # Retourne seulement les prédictions
    """
    Applique les masques RISE à une image, obtient les prédictions du modèle pour les images masquées.
    Retourne les prédictions (probabilités softmax).
    """
    if chaine_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(chaine_device)

    modele.to(device)
    modele.eval()

    if tenseur_image_originale.ndim == 3: # (C, H, W)
        tenseur_image_originale = tenseur_image_originale.unsqueeze(0) # Ajouter dim batch : (1, C, H, W)
    
    tenseur_image_originale = tenseur_image_originale.to(device)
    tenseur_masques_rise = tenseur_masques_rise.to(device) # Déplacer les masques aussi

    nb_masques = tenseur_masques_rise.shape[0]
    
    liste_toutes_predictions = []
    
    print(f"Application de {nb_masques} masques et prédiction par lots de {taille_batch_preds}...")
    with torch.no_grad():
        for i in tqdm(range(0, nb_masques, taille_batch_preds)):
            batch_masques = tenseur_masques_rise[i : i + taille_batch_preds] # (batch, H, W)
            
            # Multiplication élément par élément : image_originale (1,C,H,W) * batch_masques (batch,1,H,W)
            # Décompresser batch_masques pour le rendre diffusable avec les canaux de l'image
            images_masquees_batch_actuel = tenseur_image_originale * batch_masques.unsqueeze(1) # (batch, C, H, W)
            
            # Obtenir les prédictions pour le lot actuel d'images masquées
            logits = modele(images_masquees_batch_actuel) # (batch, NbClasses)
            probabilities = F.softmax(logits, dim=1)    # (batch, NbClasses)
            liste_toutes_predictions.append(probabilities.cpu()) # Collecter sur CPU pour économiser mémoire GPU

    tenseur_toutes_predictions = torch.cat(liste_toutes_predictions, dim=0)
    
    return tenseur_toutes_predictions # Forme (NbMasques, NbClasses)
```

## def creer_carte_saliency_rise(tenseur_masques_rise: torch.Tensor, tenseur_predictions: torch.Tensor, index_classe: int) -> torch.Tensor
Crée une carte de saillance RISE pour une classe spécifique en effectuant une somme pondérée des masques RISE.
Les poids sont les scores de prédiction du modèle (probabilités) pour la classe spécifiée pour chaque image masquée.
Extrait du TP7 (Partie Optionnelle).

**Entrées :**
- `tenseur_masques_rise` (torch.Tensor) : Tenseur des masques RISE, forme (NbMasques, H, W).
- `tenseur_predictions` (torch.Tensor) : Tenseur des prédictions du modèle (probabilités softmax) pour les images masquées,
forme (NbMasques, NbClasses). C'est la sortie de `appliquer_masques_rise_et_predire`.
- `index_classe` (int) : L'index de la classe cible pour laquelle générer la carte de saillance.

**Sorties :**
- `carte_saliency` (torch.Tensor) : La carte de saillance générée, forme (H, W), normalisée à [0, 1].

```python
def creer_carte_saliency_rise(
    tenseur_masques_rise: torch.Tensor,     # (NbMasques, H, W)
    tenseur_predictions: torch.Tensor,  # (NbMasques, NbClasses)
    index_classe: int
) -> torch.Tensor:
    """
    Crée une carte de saillance RISE pour une classe spécifique.
    C'est une somme pondérée de masques, où les poids sont les scores de prédiction pour cette classe.
    """
    if index_classe < 0 or index_classe >= tenseur_predictions.shape[1]:
        raise ValueError(f"index_classe invalide {index_classe}. Doit être entre 0 et {tenseur_predictions.shape[1]-1}.")

    # Obtenir les scores de prédiction pour l'index_classe cible pour toutes les images masquées
    # Ces scores seront les poids pour les masques
    poids_pour_masques = tenseur_predictions[:, index_classe] # Forme : (NbMasques)

    # S'assurer que tenseur_masques_rise et poids_pour_masques sont sur le même device (ex: CPU pour cela)
    # Et s'assurer qu'ils sont float pour la multiplication
    masques = tenseur_masques_rise.cpu().float()
    poids = poids_pour_masques.cpu().float()

    # Somme pondérée des masques : Somme_i ( masque_i * poids_i )
    # masques est (NbMasques, H, W), poids est (NbMasques)
    # Il faut redimensionner poids en (NbMasques, 1, 1) pour le diffuser avec les masques
    
    # Multiplication élément par élément et somme sur la dimension NbMasques
    # carte_saliency = torch.sum(masques * poids.view(-1, 1, 1), dim=0)
    # Alternative utilisant einsum pour la clarté
    carte_saliency = torch.einsum('n h w, n -> h w', masques, poids)

    # Normaliser la carte de saillance pour qu'elle soit dans [0, 1] pour la visualisation
    min_val = torch.min(carte_saliency)
    max_val = torch.max(carte_saliency)
    
    if max_val > min_val: # Éviter la division par zéro si la carte est plate
        carte_saliency_normalisee = (carte_saliency - min_val) / (max_val - min_val)
    else: # Si la carte est plate (ex: tout zéros si tous les poids étaient zéro)
        carte_saliency_normalisee = torch.zeros_like(carte_saliency) 

    return carte_saliency_normalisee # Forme : (H, W)
```

## def visualiser_resultats_saliency_rise(image_originale_pil: Image.Image, cartes_saliency_par_classe: dict, titre_principal: str = "Cartes de Saillance RISE")
Trace l'image originale aux côtés des cartes de saillance RISE pour plusieurs classes.
Chaque carte de saillance est superposée sur une version en niveaux de gris de l'image originale.
Extrait du TP7 (Partie Optionnelle).

**Entrées :**
- `image_originale_pil` (PIL.Image.Image) : L'image d'entrée originale sous forme d'objet PIL Image.
- `cartes_saliency_par_classe` (dict) : Un dictionnaire où les clés sont des noms de classes (str) et
les valeurs sont les cartes de saillance correspondantes (torch.Tensor, forme (H,W)).
- `titre_principal` (str, optionnel) : Titre général de la figure. Par défaut "Cartes de Saillance RISE".

**Sorties :**
- Aucune (affiche le graphique).

```python
def visualiser_resultats_saliency_rise(image_originale_pil: Image.Image, cartes_saliency_par_classe: dict, titre_principal: str = "Cartes de Saillance RISE"):
    """
    Trace l'image originale et les superpositions de cartes de saillance RISE pour plusieurs classes.
    image_originale_pil: Image PIL de l'entrée originale.
    cartes_saliency_par_classe: Dict {'nom_classe': tenseur_carte_saliency_HW, ...}
    """
    nb_classes_a_tracer = len(cartes_saliency_par_classe)
    
    if nb_classes_a_tracer == 0:
        print("Aucune carte de saillance fournie pour le traçage.")
        return

    # Créer une figure avec 1 ligne et (1 + nb_classes_a_tracer) colonnes
    # La première colonne pour l'image originale, les suivantes pour les cartes de saillance
    fig, axes = plt.subplots(1, 1 + nb_classes_a_tracer, figsize=((1 + nb_classes_a_tracer) * 4.5, 4.5)) # Ajuster taille
    fig.suptitle(titre_principal, fontsize=16)

    # Si une seule classe est tracée, axes n'est pas un tableau, le rendre compatible
    if nb_classes_a_tracer == 0: # Devrait être géré par le return ci-dessus, mais pour la robustesse
        axes_iterable = [axes]
    elif not isinstance(axes, np.ndarray): # Cas d'un seul subplot (1 image + 0 carte ou 1 image + 1 carte)
        axes_iterable = np.array([axes]) # Le rendre itérable
    else:
        axes_iterable = axes.flatten()


    # 1. Afficher l'image originale
    axes_iterable[0].imshow(image_originale_pil)
    axes_iterable[0].set_title("Image Originale")
    axes_iterable[0].axis('off')

    # Convertir l'image PIL originale en tableau numpy niveaux de gris pour le fond de la superposition
    img_originale_gris_np = np.array(image_originale_pil.convert('L'), dtype=np.float32) / 255.0 # Niveaux de gris [0,1]

    # 2. Afficher les superpositions de cartes de saillance pour chaque classe
    idx_graph = 1
    for nom_classe, tenseur_carte_saliency in cartes_saliency_par_classe.items():
        if idx_graph >= len(axes_iterable): break # Sécurité si moins d'axes que prévu

        np_carte_saliency = tenseur_carte_saliency.cpu().numpy() # S'assurer que c'est un tableau NumPy (H, W)
        
        ax = axes_iterable[idx_graph]
        ax.imshow(img_originale_gris_np, cmap='gray') # Fond
        # Superposer la carte de saillance. 'jet' est une palette de couleurs courante pour les heatmaps.
        im = ax.imshow(np_carte_saliency, cmap='jet', alpha=0.6) 
        ax.set_title(f"RISE pour : {nom_classe}")
        ax.axis('off')
        # Optionnel : Ajouter une barre de couleurs pour la carte de saillance
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Ajuster les paramètres au besoin
        idx_graph += 1
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajuster pour le suptitle
    plt.show()

```
