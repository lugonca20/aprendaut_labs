import numpy as np
import pandas as pd

class NaiveBayes:
    """Naive Bayes categórico multiclase con suavizado m-estimate."""

    def __init__(self, m=1.0):
        self.m = m

    def fit(self, X, y):
        self.clases_ = sorted(y.unique())
        cantidad_total = len(y)

        self.log_prioris_ = {}
        self.tablas_categoricas_ = {}    # tablas_categoricas_[clase][columna][valor] = probabilidad
        self.respaldo_categoricas_ = {}  # probabilidad para valores no vistos en entrenamiento
        self.dominio_columnas_ = {columna: X[columna].unique().tolist() for columna in X.columns}

        for clase in self.clases_:
            mascara_clase = (y == clase)
            X_clase = X.loc[mascara_clase]
            cantidad_clase = mascara_clase.sum()
            self.log_prioris_[clase] = np.log(cantidad_clase / cantidad_total)

            self.tablas_categoricas_[clase] = {}
            self.respaldo_categoricas_[clase] = {}
            for columna in X.columns:
                dominio = self.dominio_columnas_[columna]
                probabilidad_uniforme = 1.0 / len(dominio)
                conteos = X_clase[columna].value_counts()

                tabla_columna = {
                    valor: (conteos.get(valor, 0) + self.m * probabilidad_uniforme) / (cantidad_clase + self.m)
                    for valor in dominio
                }
                self.tablas_categoricas_[clase][columna] = tabla_columna
                # Probabilidad para un valor jamás visto en entrenamiento (conteo = 0)
                self.respaldo_categoricas_[clase][columna] = (self.m * probabilidad_uniforme) / (cantidad_clase + self.m)

        return self

    # evitar log(0) usando la probabilidad de respaldo para valores no vistos
    def _log_verosimilitud(self, fila, clase):
        log_probabilidad = 0.0
        for columna, tabla_columna in self.tablas_categoricas_[clase].items():
            valor = fila[columna]
            log_probabilidad += np.log(tabla_columna.get(valor, self.respaldo_categoricas_[clase][columna]))
        return log_probabilidad

    def predict_log_proba(self, X):
        filas = [
            {
                clase: self.log_prioris_[clase] + self._log_verosimilitud(fila, clase)
                for clase in self.clases_
            }
            for _, fila in X.iterrows()
        ]
        return pd.DataFrame(filas, index=X.index)[self.clases_]

    def predict(self, X):
        log_probabilidades = self.predict_log_proba(X)
        return log_probabilidades.idxmax(axis=1)
