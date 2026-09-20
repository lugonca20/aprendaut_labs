import pandas as pd
from pandas import DataFrame

def construir_historial_equipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Este Dataframe permite calcular estadísticas de cualquier equipo con un único
    groupby("team"), sin importar si jugó de local o visitante.
    """
    home_df = pd.DataFrame({
        "match_id":        df["match_id"],
        "date":            df["date"],
        "team":            df["home"],
        "year":            df["year"],
        "is_home":         True,
        "goles_anotados":  df["gh"],
        "goles_recibidos": df["ga"],
        "win":             (df["ganador"] == "L").astype(int),
    })
    away_df = pd.DataFrame({
        "match_id":        df["match_id"],
        "date":            df["date"],
        "team":            df["away"],
        "year":            df["year"],
        "is_home":         False,
        "goles_anotados":  df["ga"], 
        "goles_recibidos": df["gh"], 
        "win":             (df["ganador"] == "V").astype(int),
    })

    historial_equipos_df = pd.concat([home_df, away_df], ignore_index=True)
    return historial_equipos_df.sort_values(["year", "team", "date"]).reset_index(drop=True)


def _calcular_atributos_totales(historial_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calcula victorias y goles (anotados y recibidos) en los ultimos N (window) partidos tanto
    de local como de visitante y días de descanso desde el ultimo partido para cada equipo
    considerando únicamente los partidos de una misma temporada o año.

    El shift(1) excluye el partido actual para evitar data leakage.
    """
    historial_df = historial_df.copy()

    for dest_col, src_col in [
        (f"wins_last_{window}",            "win"),
        (f"goals_scored_last_{window}",    "goles_anotados"),
        (f"goals_conceded_last_{window}",  "goles_recibidos"),
    ]:
        historial_df[dest_col] = (
            historial_df.groupby(["year", "team"])[src_col]
            .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum())
            .fillna(0)
            .astype(int)
        )

    # Días de descanso: diferencia entre la fecha del partido actual
    # y la fecha del partido anterior del mismo equipo indenpendientemente de su condición (local o visitante).
    # 90 si el equipo no tiene historial previo.
    prev_dates = historial_df.groupby(["year", "team"])["date"].transform(lambda x: x.shift(1))
    historial_df["dias_descanso"] = (
        (historial_df["date"] - prev_dates).dt.days
        .fillna(90)
        .astype(int)
    )

    return historial_df


def _calcular_atributos_condicionales(historial_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Agrega victorias en los últimos N partidos de Local y de Visitante,
    calculadas por separado.
    """
    historial_df = historial_df.copy()

    for is_home_flag, col in [
        (True,  f"home_wins_last_{window}"),
        (False, f"away_wins_last_{window}"),
    ]:
        # Filtrar sólo los partidos del venue correspondiente
        subset = historial_df[historial_df["is_home"] == is_home_flag][["match_id", "year", "team", "win"]].copy()

        # Ventana rodante sobre ese venue únicamente
        subset[col] = (
            subset.groupby(["year", "team"])["win"]
            .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum())
            .fillna(0)
            .astype(int)
        )

        # Merge de vuelta: filas del venue opuesto quedan en 0
        historial_df = historial_df.merge(
            subset[["match_id", "team", col]],
            on=["match_id", "team"],
            how="left",
        )
        # Relleno con ceros las filas en las que el equipo no cumple con la condición (Local o Visitante)
        historial_df[col] = historial_df[col].fillna(0).astype(int)

    return historial_df

def _rename_columns(historial_df: pd.DataFrame, window: int, is_home_flag: bool, key: str, pfx: str) -> pd.DataFrame:
    return (
            historial_df[historial_df["is_home"] == is_home_flag][[
                "match_id", "team",
                f"wins_last_{window}",
                f"{key}_wins_last_{window}",
                f"goals_scored_last_{window}",
                f"goals_conceded_last_{window}",
                "dias_descanso",
            ]]
            .rename(columns={
                "team":                          key,
                f"wins_last_{window}":           f"{pfx}_wins_last_{window}",
                f"goals_scored_last_{window}":   f"{pfx}_goals_scored_last_{window}",
                f"goals_conceded_last_{window}": f"{pfx}_goals_conceded_last_{window}",
                "dias_descanso":                 f"{pfx}_dias_descanso",
            })
        )

def calculate_historic_features(dataset: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Agrega atributos históricos al dataset.

    Columnas generadas:
        Prefijo 'local_'  → equipo que juega de local en este partido
        Prefijo 'visita_' → equipo que juega de visitante en este partido

        {pfx}_wins_last_{N}                  Victorias en últimos N partidos (de Local o de Visitante)
        {pfx}_goals_scored_last_{N}     Goles anotados en últimos N partidos (de Local o de Visitante)
        {pfx}_goals_conceded_last_{N}  Goles recibidos en últimos N partidos (de Local o de Visitante)
        {pfx}_dias_descanso                      Días desde el último partido (90 si no hay historial)
        away_wins_last_{N}                              Victorias en últimos N partidos como Visitante
        home_wins_last_{N}                                  Victorias en últimos N partidos como Local
    """
    df = dataset.sort_values("date").reset_index(drop=True).copy()
    df["match_id"] = df.index

    historial_df = construir_historial_equipos(df)
    historial_df = _calcular_atributos_totales(historial_df, window)
    historial_df = _calcular_atributos_condicionales(historial_df, window)

    home_stats = _rename_columns(historial_df, window, True, "home", "local")
    away_stats = _rename_columns(historial_df, window, False, "away", "visita")

    df_out = df.merge(home_stats, on=["match_id", "home"], how="left")
    df_out = df_out.merge(away_stats, on=["match_id", "away"], how="left")

    return df_out.drop(columns=["match_id"])