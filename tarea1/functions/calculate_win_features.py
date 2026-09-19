import pandas as pd

from functions.historic_features import construir_historial_equipos


def calculate_win_features(dataset: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Agrega la diferencia de victorias recientes entre el local y el visitante.

    Para cada equipo se cuentan las victorias en sus últimos N (window) partidos de la
    misma temporada, sin importar si los jugó de local o de visitante, y se resta la
    cantidad del visitante a la del local.

    El shift(1) excluye el partido actual para evitar data leakage.

    Columnas generadas:
        dif_wins_last_{N}     Positiva si el local viene ganando más que el visitante,
                              negativa si es al revés y 0 si vienen parejos.

    Debe aplicarse antes de eliminar 'gh', 'ga' y 'date', ya que se utilizan para
    construir el historial de cada equipo.
    """
    df = dataset.sort_values("date").reset_index(drop=True).copy()
    df["match_id"] = df.index

    historial_df = construir_historial_equipos(df)
    historial_df[f"wins_last_{window}"] = (
        historial_df.groupby(["year", "team"])["win"]
        .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )

    home_stats = (
        historial_df[historial_df["is_home"]][["match_id", f"wins_last_{window}"]]
        .rename(columns={f"wins_last_{window}": "local_wins"})
    )
    away_stats = (
        historial_df[~historial_df["is_home"]][["match_id", f"wins_last_{window}"]]
        .rename(columns={f"wins_last_{window}": "visita_wins"})
    )

    df_out = df.merge(home_stats, on="match_id", how="left")
    df_out = df_out.merge(away_stats, on="match_id", how="left")
    df_out[f"dif_wins_last_{window}"] = df_out["local_wins"] - df_out["visita_wins"]

    return df_out.drop(columns=["match_id", "local_wins", "visita_wins"])
