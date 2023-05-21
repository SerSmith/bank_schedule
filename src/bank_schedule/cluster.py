"""Скрипты для кластеризации банкоматов
"""
from typing import Optional
import pandas as pd

from sklearn.cluster import KMeans
from bank_schedule.data import Data
from bank_schedule import helpers
from bank_schedule.constants import RS, RAW_DATA_FOLDER

LABEL_COL = 'label'


def clusterize_atm(loader: Data,
                   n_clusters: int,
                   random_state: Optional[int]=RS) -> pd.DataFrame:
    """Распределяет банкоматы по кластерам с помощью KMeans

    Args:
        loader (Data): объект bank_schedule.data.Data
        n_clusters (int): _description_
        random_state (Optional[int], optional): _description_. Defaults to RS.

    Returns:
        pd.DataFrame: датафрейм с колонками ['TID', 'label']
    """
    # load geodata
    geo_df = loader.get_geo_TIDS()

    # get cartesian coordinates
    coords_df = helpers.calc_cartesian_coords(lat_series=geo_df['latitude'],
                                              lon_series=geo_df['longitude'])
    coords_df = geo_df.join(coords_df)

    # clusterize
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init='auto',
                    random_state=random_state)

    labels = kmeans.fit_predict(coords_df[['x', 'y']], sample_weight=None)

    coords_df[LABEL_COL] = labels

    return coords_df[['TID', LABEL_COL]]


if __name__ == '__main__':
    my_loader = Data(RAW_DATA_FOLDER)
    print(clusterize_atm(my_loader, n_clusters=50))
