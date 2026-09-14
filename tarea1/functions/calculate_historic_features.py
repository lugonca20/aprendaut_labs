import pandas as pd


def calculate(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.sort_values(by="date").reset_index(drop=True)
    df["match_id"] = df.index

    home_df = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "year": df["year"],
            "team": df["home"],
            "goles_anotados": df["gh"],
            "goles_recibidos": df["ga"],
        }
    )

    away_df = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "year": df["year"],
            "team": df["away"],
            "goles_anotados": df["ga"],
            "goles_recibidos": df["gh"],
        }
    )

    historial_equipos = pd.concat([home_df, away_df], ignore_index=True)
    historial_equipos = historial_equipos.sort_values(["year", "team", "date"])

    historial_equipos["racha_goles_anotados"] = (
        historial_equipos.groupby(["year", "team"])["goles_anotados"]
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).sum())
        .fillna(0)
    )

    historial_equipos["racha_goles_recibidos"] = (
        historial_equipos.groupby(["year", "team"])["goles_recibidos"]
        .transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).sum())
        .fillna(0)
    )

    historial_equipos["fecha_ultimo_partido"] = historial_equipos.groupby(
        ["year", "team"]
    )["date"].transform(lambda x: x.shift(1))

    historial_equipos["dias_descanso"] = (
        historial_equipos["date"] - historial_equipos["fecha_ultimo_partido"]
    ).dt.days

    historial_equipos["dias_descanso"] = (
        historial_equipos["dias_descanso"].fillna(99).astype(int)
    )

    home_stats = historial_equipos.merge(
        df[["match_id", "home"]],
        left_on=["match_id", "team"],
        right_on=["match_id", "home"],
    )[
        ["match_id", "racha_goles_anotados", "racha_goles_recibidos", "dias_descanso"]
    ].rename(
        columns={
            "racha_goles_anotados": "racha_goles_anotados_home",
            "racha_goles_recibidos": "racha_goles_recibidos_home",
            "dias_descanso": "dias_descanso_home",
        }
    )

    away_stats = historial_equipos.merge(
        df[["match_id", "away"]],
        left_on=["match_id", "team"],
        right_on=["match_id", "away"],
    )[
        ["match_id", "racha_goles_anotados", "racha_goles_recibidos", "dias_descanso"]
    ].rename(
        columns={
            "racha_goles_anotados": "racha_goles_anotados_away",
            "racha_goles_recibidos": "racha_goles_recibidos_away",
            "dias_descanso": "dias_descanso_away",
        }
    )

    df_out = df.merge(home_stats, on="match_id", how="left")
    df_out = df_out.merge(away_stats, on="match_id", how="left")

    return df_out.drop(columns=["match_id"])