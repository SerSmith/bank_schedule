"""Скрипты для кластеризации банкоматов
"""
from typing import Optional
import pandas as pd

from k_means_constrained import KMeansConstrained

from bank_schedule.data import Data
from bank_schedule import helpers
from bank_schedule.constants import RS, RAW_DATA_FOLDER

LABEL_COL = 'label'


def clusterize_atm(loader: Data,
                   n_clusters: int,
                   allowed_percent=None,
                   random_state: Optional[int]=RS) -> pd.DataFrame:
    """Распределяет банкоматы по кластерам с помощью KMeansConstrained

    Args:
        loader (Data): объект bank_schedule.data.Data
        n_clusters (int): xbckj число кластеров
        allowed_percent (float)L допустимые отклонения от среднего в размере кластерв
        (count(tid) / n_clusters) * (size_max) должно быть больше или равно числу точек в данных
        random_state (Optional[int], optional): random_state. Defaults to RS.

    Returns:
        pd.DataFrame: датафрейм с колонками ['TID', 'label']
    """

    # load geodata
    geo_df = loader.get_geo_TIDS()

    size_min = None
    size_max = None

    if allowed_percent is not None:
        quant = geo_df["TID"].count() / n_clusters
        size_min = round(max(quant * (1 - allowed_percent), 1))
        size_max = round(quant * (1 + allowed_percent))



    # get cartesian coordinates
    coords_df = helpers.calc_cartesian_coords(lat_series=geo_df['latitude'],
                                              lon_series=geo_df['longitude'])
    coords_df = geo_df.join(coords_df)

    # clusterize
    # kmeans = KMeans(n_clusters=n_clusters,
    #                 n_init='auto',
    #                 random_state=random_state)


    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state
    )

    labels = kmeans.fit_predict(coords_df[['x', 'y']])

    coords_df[LABEL_COL] = labels

    return coords_df[['TID', LABEL_COL]]


if __name__ == '__main__':
    my_loader = Data(RAW_DATA_FOLDER)
    print(clusterize_atm(my_loader,
                         n_clusters=10,
                         allowed_percent=0.01))

