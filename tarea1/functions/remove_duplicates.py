import pandas as pd

def remove(dt: pd.DataFrame) -> pd.DataFrame:
    filas_antes = dt.shape[0]
    dt.drop_duplicates(inplace=True)
    filas_despues = dt.shape[0]

    print(f"Duplicados eliminados: {filas_antes - filas_despues}")
    print(f"Instancias restantes: {filas_despues}")
    return dt