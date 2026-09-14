import pandas as pd

def remove(dt: pd.DataFrame) -> pd.DataFrame:

    home_name_to_id = dt.groupby("home")["home_ident"].nunique(dropna=False)
    home_id_to_name = dt.groupby("home_ident")["home"].nunique(dropna=False)

    away_name_to_id = dt.groupby("away")["away_ident"].nunique(dropna=False)
    away_id_to_name = dt.groupby("away_ident")["away"].nunique(dropna=False)

    print(f"Cada valor de home se corresponde a un solo valor de home_ident, y viceversa: ", (home_name_to_id == 1).all()  & (home_id_to_name == 1).all())
    print(f"Cada valor de away se corresponde a un solo valor de away_ident, y viceversa: ", (away_name_to_id == 1).all()  & (away_id_to_name == 1).all())

    dt = dt.drop(columns=['home_ident', 'away_ident'])

    return dt