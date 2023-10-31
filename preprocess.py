import pandas as pd
import ftfy
from typing import Tuple


def preprocess_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    apply some data cleaning and split the input data into train, test and dev sets
    :param df: input dataframe
    :return: cleaned dataframe split into train, test and dev
    """
    df["headline"] = df["headline"].apply(lambda x: ftfy.fix_text(x))
    train = df[:int(len(df) * 0.8)]
    test = df[int(len(df) * 0.8): int(len(df) * 0.9)]
    dev = df[int(len(df) * 0.9):]
    return train, test, dev
