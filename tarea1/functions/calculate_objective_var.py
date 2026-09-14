import pandas as pd
import numpy as np

def calculate(dt: pd.DataFrame) -> pd.DataFrame:
    dt["ganador"] = np.select(
        [
            dt["gh"] > dt["ga"],     
            dt["gh"] < dt["ga"],
            dt["gh"] == dt["ga"],
        ],
        [
            "L",
            "V",
            "E",
        ],
        default="sin_dato",
    )
    return dt