import pandas as pd
from pathlib import Path

DATASETS_DIR = 'datasets'


def load_Y(dataset_name, as_numpy=True):
    path = Path('shrinkage') / DATASETS_DIR / f'{dataset_name}.csv'
    Y = pd.read_csv(path, index_col=0)
    if as_numpy:
        Y = Y.values
    return Y


# run this once (py -m datasets)
# in order to create some well-defined datasets
# from ones available in another form
# (unless they already exist)
if __name__=='__main__':

    # S&P 500
    Y = pd.read_csv(Path(DATASETS_DIR) / 'sp500_5yr_pre.csv')
    Y = Y.pivot(
        index='Name',
        columns='date',
        values='close'
    )
    Y = Y.dropna()
    Y = Y.pct_change(axis=1).dropna(axis=1)
    Y.to_csv(Path(DATASETS_DIR) / 'sp500_5yr.csv')