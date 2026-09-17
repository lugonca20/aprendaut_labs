from tarea1.functions.historic_features import construir_historial_equipos
import pandas as pd

def build_snapshot(df_train: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Construye el estado de cada equipo al final del set de entrenamiento.
    Usar para predecir partidos futuros no incluidos en df_train.

    Equivalente simétrico de calculate_historic_features para inferencia:
    toma los últimos `window` partidos reales de cada equipo y los agrega.
    No usa shift(1) porque no hay partido "actual" que excluir.

    Retorna un DataFrame con una fila por equipo:
        team
        wins_last{N}            Victorias en últimos N partidos (cualquier venue)
        home_wins_last{N}       Victorias en últimos N partidos como local
        away_wins_last{N}       Victorias en últimos N partidos como visitante
        goals_scored_last{N}    Goles anotados en últimos N partidos (cualquier venue)
        goals_conceded_last{N}  Goles recibidos en últimos N partidos (cualquier venue)
        fecha_ultimo_partido    Fecha del último partido jugado (para calcular dias_descanso)

    Ejemplo de uso:
        snapshot = build_team_snapshot(df_train, window=10)

        # Calcular días de descanso para una fecha futura
        fecha_partido = pd.Timestamp("2025-05-01")
        snapshot["dias_descanso"] = (
            fecha_partido - snapshot["fecha_ultimo_partido"]
        ).dt.days

        # Aplicar a un partido futuro
        partido_futuro = pd.DataFrame([{"home": "Team A", "away": "Team B"}])

        feats_local = snapshot.rename(columns={
            **{c: f"local_{c}" for c in snapshot.columns if c != "team"},
            "team": "home",
        })
        feats_visita = snapshot.rename(columns={
            **{c: f"visita_{c}" for c in snapshot.columns if c != "team"},
            "team": "away",
        })

        resultado = (
            partido_futuro
            .merge(feats_local,  on="home", how="left")
            .merge(feats_visita, on="away", how="left")
        )
    """
    df = df_train.sort_values("date").reset_index(drop=True).copy()
    df["match_id"] = df.index

    hist = construir_historial_equipos(df)
    hist = hist.sort_values(["team", "date"])

    # Últimos N partidos por equipo (todos los venues)
    last_n_all = hist.groupby("team").tail(window)

    agg_dict = {
        "fecha_ultimo_partido":        ("date",            "max"),
        f"wins_last{window}":          ("win",             "sum"),
        f"goals_scored_last{window}":  ("goles_anotados",  "sum"),
        f"goals_conceded_last{window}":("goles_recibidos", "sum"),
    }
    snapshot = last_n_all.groupby("team", as_index=False).agg(**agg_dict)

    # Victorias por venue (ventana separada de home-only / away-only)
    for is_home_flag, col in [
        (True,  f"home_wins_last{window}"),
        (False, f"away_wins_last{window}"),
    ]:
        last_n_venue = (
            hist[hist["is_home"] == is_home_flag]
            .groupby("team")
            .tail(window)
        )
        venue_wins = (
            last_n_venue
            .groupby("team")["win"]
            .sum()
            .reset_index(name=col)
        )
        snapshot = snapshot.merge(venue_wins, on="team", how="left")
        snapshot[col] = snapshot[col].fillna(0).astype(int)

    return snapshot
