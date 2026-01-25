import numpy as np
import pandas as pd
import os

def import_raw_from_csv(dir_path: str) -> pd.DataFrame:
    df_train = pd.read_csv(f'{dir_path}/train.csv')
    df_test = pd.read_csv(f'{dir_path}/test.csv')
    return df_train, df_test

def date_raw_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['Time'] = (
    pd.to_datetime(df['Date']) - pd.Timestamp("1970-01-01")
    ).dt.days
    return df
    # return df.drop(columns = 'Date')

def write_predictions_csv(
    pred: list[float],
    sample_pred_path: str,
    output_path: str
):
    submit = pd.read_csv(sample_pred_path)
    submit["Net_demand"] = pred
    submit.to_csv(output_path, index=False)

