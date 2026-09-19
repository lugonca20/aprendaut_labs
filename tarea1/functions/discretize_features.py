import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from functions.historic_features import construir_historial_equipos

# Ventanas con las que se calcularon los atributos históricos
WINDOW_GOLES = 10
WINDOW_WINS = 5

CORTES_GOLES = [-np.inf, 5, 9, 13, np.inf]
CORTES_GOLES_PROM = [-np.inf, 0.8, 1.1, 1.5, np.inf]
ETIQUETAS_GOLES = ["muy_baja", "baja", "media", "alta"]

# 90 es el valor por defecto de historic_features para el primer partido de la temporada
CORTES_DESCANSO = [-np.inf, 3, 6, 7, 14, 89, np.inf]
ETIQUETAS_DESCANSO = ["<=3", "4-6", "7", "8-14", ">14", "primer_partido"]

CORTES_DIF_WINS = [-np.inf, -2.5, -0.5, 0.5, 2.5, np.inf]
ETIQUETAS_DIF_WINS = [
    "visitante mucho mejor",
    "visitante mejor",
    "parejos",
    "local mejor",
    "local mucho mejor",
]

COLS_GOLES = [
    f"local_goals_scored_last_{WINDOW_GOLES}",
    f"local_goals_conceded_last_{WINDOW_GOLES}",
    f"visita_goals_scored_last_{WINDOW_GOLES}",
    f"visita_goals_conceded_last_{WINDOW_GOLES}",
]
COLS_DESCANSO = ["local_dias_descanso", "visita_dias_descanso"]
COLS_DIF_WINS = [f"dif_wins_last_{WINDOW_WINS}"]


def calculate_partidos_en_ventana(dataset: pd.DataFrame, window: int = WINDOW_GOLES) -> pd.DataFrame:
    """
    Agrega la cantidad de partidos previos de la temporada que entran en la ventana
    de los atributos históricos (como máximo N = window).

    Se utiliza para normalizar los goles: por el min_periods=1 del rolling, al inicio de
    la temporada los goles se suman sobre menos partidos y una racha "baja" refleja la
    etapa del campeonato más que la fuerza del equipo.

    Columnas generadas:
        local_partidos_en_ventana      Partidos previos del equipo local (máximo N)
        visita_partidos_en_ventana     Partidos previos del equipo visitante (máximo N)

    Debe aplicarse antes de eliminar 'gh', 'ga' y 'date'.
    """
    df = dataset.sort_values("date").reset_index(drop=True).copy()
    df["match_id"] = df.index

    historial_df = construir_historial_equipos(df)
    historial_df["partidos_en_ventana"] = (
        historial_df.groupby(["year", "team"]).cumcount().clip(upper=window)
    )

    df_out = df
    for is_home_flag, pfx in [(True, "local"), (False, "visita")]:
        stats = (
            historial_df[historial_df["is_home"] == is_home_flag][["match_id", "partidos_en_ventana"]]
            .rename(columns={"partidos_en_ventana": f"{pfx}_partidos_en_ventana"})
        )
        df_out = df_out.merge(stats, on="match_id", how="left")

    return df_out.drop(columns=["match_id"])


class DiscretizadorHistorico(BaseEstimator, TransformerMixin):
    """
    Discretiza los atributos históricos en intervalos de cortes fijos. Cada valor se
    reemplaza por el índice del intervalo al que pertenece, de modo que se conserva el
    orden entre intervalos.

    Si normalizar_goles=True, los goles se dividen por la cantidad de partidos en la
    ventana antes de discretizar (requiere calculate_partidos_en_ventana).
    """

    def __init__(self, normalizar_goles=False):
        self.normalizar_goles = normalizar_goles

    def fit(self, X, y=None):
        # Los cortes son fijos, no se estiman a partir de los datos
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in COLS_GOLES:
            if col not in X.columns:
                continue
            valores = X[col]
            cortes = CORTES_GOLES
            if self.normalizar_goles:
                pfx = col.split("_")[0]
                ventana = X[f"{pfx}_partidos_en_ventana"].clip(lower=1)
                valores = valores / ventana
                cortes = CORTES_GOLES_PROM
            X[col] = pd.cut(valores, cortes, labels=False).astype("int8")

        for col in COLS_DESCANSO:
            if col in X.columns:
                X[col] = pd.cut(X[col], CORTES_DESCANSO, labels=False).astype("int8")

        for col in COLS_DIF_WINS:
            if col in X.columns:
                X[col] = pd.cut(X[col], CORTES_DIF_WINS, labels=False).astype("int8")

        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def etiquetas(col: str) -> list:
    if col in COLS_GOLES:
        return ETIQUETAS_GOLES
    if col in COLS_DESCANSO:
        return ETIQUETAS_DESCANSO
    if col in COLS_DIF_WINS:
        return ETIQUETAS_DIF_WINS
    raise ValueError(f"{col} no es un atributo discretizado")


def etiquetar(X: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza el índice de cada intervalo por su etiqueta. Solo para inspeccionar los
    datos: dentro del Pipeline las etiquetas se volverían categóricas y el OneHotEncoder
    perdería el orden de los intervalos.
    """
    X = pd.DataFrame(X).copy()
    for col in COLS_GOLES + COLS_DESCANSO + COLS_DIF_WINS:
        if col in X.columns:
            X[col] = X[col].map(dict(enumerate(etiquetas(col))))
    return X
