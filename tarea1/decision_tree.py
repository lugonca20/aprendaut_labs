import pandas as pd
import numpy as np

class NodoArbol:
    """Nodo de un árbol de decisión: hoja con etiqueta, o nodo interno con una regla de split."""

    def __init__(self):
        self.es_hoja = False
        self.etiqueta = None   # clase mayoritaria: resultado si es hoja, respaldo si no
        self.atributo = None
        self.hijos = {}         # {valor del atributo: nodo hijo}


class ArbolDecisionID3():
    """
    Árbol de decisión ID3 puro: todos los atributos se tratan como categóricos,
    con split multi-vía (una rama por valor distinto) y ganancia de información
    como criterio de split. min_info_gain es el criterio de parada de la recursión.
    """

    def __init__(self, min_info_gain=0.0, max_depth=None):
        self.min_info_gain = min_info_gain
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.classes_ = np.unique(y)

        atributos = list(X.columns)
        self.raiz_ = self._construir(X, y, atributos, profundidad=0)
        return self

    def predict(self, X: pd.DataFrame):
        return X.apply(lambda fila: self._predecir_fila(fila, self.raiz_), axis=1).to_numpy()

    def _construir(self, X: pd.DataFrame, y: pd.Series, atributos: list, profundidad: int) -> NodoArbol:
        nodo = NodoArbol()
        nodo.etiqueta = y.mode()[0]

        if y.nunique() == 1:
            nodo.es_hoja = True
            return nodo
        if not atributos:
            nodo.es_hoja = True
            return nodo
        if self.max_depth is not None and profundidad >= self.max_depth:
            nodo.es_hoja = True
            return nodo

        entropia_actual = self._entropia(y)
        mejor = self._elegir_mejor_atributo(X, y, atributos, entropia_actual)

        if mejor is None or mejor['ganancia'] <= self.min_info_gain:
            nodo.es_hoja = True
            return nodo

        nodo.atributo = mejor['atributo']

        atributos_restantes = [a for a in atributos if a != nodo.atributo]
        for valor in X[nodo.atributo].unique():
            es_valor = X[nodo.atributo] == valor
            nodo.hijos[valor] = self._construir(X[es_valor], y[es_valor], atributos_restantes, profundidad + 1)

        return nodo

    def _elegir_mejor_atributo(self, X: pd.DataFrame, y: pd.Series, atributos: list, entropia_actual: float):
        mejor = None
        for atributo in atributos:
            ganancia = self._ganancia_categorica(X[atributo], y, entropia_actual)
            if mejor is None or ganancia > mejor['ganancia']:
                mejor = {'atributo': atributo, 'ganancia': ganancia}
        return mejor

    def _entropia(self, y: pd.Series) -> float:
        if len(y) == 0:
            return 0.0
        proporciones = y.value_counts(normalize=True)
        return float(-np.sum(proporciones * np.log2(proporciones)))

    def _ganancia_categorica(self, x: pd.Series, y: pd.Series, entropia_actual: float) -> float:
        total = len(y)
        entropia_ponderada = sum(
            (cantidad / total) * self._entropia(y[x == valor])
            for valor, cantidad in x.value_counts().items()
        )
        return entropia_actual - entropia_ponderada

    def _predecir_fila(self, fila: pd.Series, nodo: NodoArbol):
        if nodo.es_hoja:
            return nodo.etiqueta

        siguiente = nodo.hijos.get(fila[nodo.atributo])
        if siguiente is None:
            return nodo.etiqueta  # valor de categoría no visto en entrenamiento

        return self._predecir_fila(fila, siguiente)