import pandas as pd


def remove(df: pd.DataFrame) -> pd.DataFrame:
    # Detectar atributos constantes (incluyendo nulos como valor único si aplica)
    constantes = df.nunique(dropna=False)[df.nunique(dropna=False) == 1]

    print("Atributos constantes detectados:")
    if constantes.empty:
        print("Ninguno.")
    else:
        print(constantes)

    df_filtered = df.loc[:, df.nunique() > 1].copy()

    return df_filtered