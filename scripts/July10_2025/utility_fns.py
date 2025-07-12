from typing import Dict
import polars as pl
from sklearn.model_selection import train_test_split


def generate_split(
    df: pl.DataFrame,
    target: str = "compressive_strength",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, tuple]:
    """
    Generates the splits for training and testing the model.
    """
    # seperate target from the regressors
    X = df.drop(target)
    y = df.select(target)
    # make th train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_pandas(), y.to_pandas(), test_size=test_size, random_state=random_state
    )
    # add an identifier -- to be used in combined
    X_train = pl.from_pandas(X_train).with_columns(partition=pl.lit("train"))
    X_test = pl.from_pandas(X_test).with_columns(partition=pl.lit("test"))
    y_train = pl.from_pandas(y_train)
    y_test = pl.from_pandas(y_test)
    # combine the two partitions
    X_combined = pl.concat([X_train, X_test], how="vertical")
    y_combined = pl.concat([y_train, y_test], how="vertical")
    # make one data
    combined = pl.concat([X_combined, y_combined], how="horizontal")
    col_order = [col for col in combined.columns if col != "partition"] + ["partition"]
    combined = combined.select(col_order)

    return {"train": (X_train, y_train), "test": (X_test, y_test), "combined": combined}


# TODO: need to put this in a unit test ---
# # test dataframe
# df = pl.DataFrame(
#     {
#         "cement": [100, 150, 120, 130, 160, 140, 110, 180],
#         "slag": [0, 20, 10, 30, 0, 25, 5, 35],
#         "ash": [10, 15, 20, 25, 10, 15, 20, 25],
#         "water": [200, 180, 190, 210, 195, 185, 200, 205],
#         "superplasticizer": [5, 6, 7, 8, 4, 5, 6, 7],
#         "compressive_strength": [40, 45, 50, 42, 47, 43, 41, 46],
#     }
# )

# df_dict = generate_split(df)
