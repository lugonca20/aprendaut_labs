from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

def load() -> pd.DataFrame:
    
    DATASET_FILE = Path("futbol_uruguayo.csv")
    if not DATASET_FILE.exists():
        DATASET_FILE = Path("tarea1/futbol_uruguayo.csv")

    assert DATASET_FILE.exists(), f"No se encontró el dataset en {DATASET_FILE}"

    dataset = pd.read_csv(DATASET_FILE)
    print(f"Cantidad inicial de instancias: {dataset.shape[0]}")
    print(f"Cantidad inicial de atributos: {dataset.shape[1]}")

    return dataset