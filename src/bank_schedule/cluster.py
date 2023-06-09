"""Скрипты для кластеризации банкоматов
"""
from typing import Optional, List
from warnings import warn
import pandas as pd

from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans

from bank_schedule.data import Data
from bank_schedule import helpers
from bank_schedule.ortools_tsp import get_best_route
from bank_schedule.constants import RS, RAW_DATA_FOLDER

LABEL_COL = 'label'


def clusterize_atm(loader: Data,
                   n_clusters: int,
                   allowed_percent=None,
                   tids_list: Optional[List[int]]=None,
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
    geo_df = loader.get_geo_TIDS().copy()

    # load geodata
    if tids_list is not None:
        geo_df = geo_df[geo_df['TID'].isin(tids_list)].reset_index(drop=True)

    size_min = None
    size_max = None

    if allowed_percent is not None and n_clusters > 1:
        quant = geo_df["TID"].count() / n_clusters
        size_min = round(max(quant * (1 - allowed_percent), 1))
        size_min = max(size_min, 1)
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


def clusterize_atm_kmeans(loader: Data,
                          n_clusters: int,
                          tids_list: Optional[List[int]]=None,
                          random_state: Optional[int]=RS) -> pd.DataFrame:
    """Распределяет банкоматы по кластерам с помощью классического KMeans

    Args:
        geo_df (pd.DataFrame): геоданные о раположении кластеров колонки [longitude, latitude],
         индекс TID
        n_clusters (int): xbckj число кластеров
        random_state (Optional[int], optional): random_state. Defaults to RS.

    Returns:
        pd.DataFrame: датафрейм с колонками ['TID', 'label']
    """

    geo_df = loader.get_geo_TIDS().copy()

    # load geodata
    if tids_list is not None:
        geo_df = geo_df[geo_df['TID'].isin(tids_list)].reset_index(drop=True)

    # get cartesian coordinates
    coords_df = helpers.calc_cartesian_coords(lat_series=geo_df['latitude'],
                                              lon_series=geo_df['longitude'])
    coords_df = geo_df.join(coords_df)

    # clusterize
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init='auto',
                    random_state=random_state)

    labels = kmeans.fit_predict(coords_df[['x', 'y']])

    coords_df[LABEL_COL] = labels

    return coords_df[['TID', LABEL_COL]]


def binary_search(tids_lis: List,
                  loader: Data,
                  min_time: float=700,
                  max_time: float=740,
                  n_iterations=1) -> int:
    """Бинарный поиск

    Args:
        array (Iterable): _description_
        elemet (Any): _description_

    Returns:
        int: _description_
    """
    if not tids_lis:
        return -1

    start, end = 0, len(tids_lis)-1
    n_it = 0

    while end > start:

        middle = int( (end + start) / 2)
        _, _, sum_time = get_best_route(loader,
                                        tids_lis[:middle],
                                        n_iterations=n_iterations)

        print(
            f'Расчетное время маршрута по кластеру после {n_it:>3} итерации: {round(sum_time, 2):>8}. '
            f'Целевое время: {round(min_time, 2)} - {round(max_time, 2)} минут'
            )

        if (sum_time >= min_time) and (sum_time <= max_time):
            return middle

        if sum_time > max_time:
            end = middle - 1
        else:
            start = middle + 1
        n_it += 1
    return end


def find_most_distant_points(dist_mtx: pd.DataFrame) -> List[int]:
    """Возвращает пару наиболее удаленных друг от друга точек

    Args:
        dist_mtx (pd.DataFrame): _description_

    Returns:
        Tuple[int, int]: _description_
    """
    sorted_df = dist_mtx.sort_values(by='Total_Time')
    return sorted_df.iloc[-1][['Origin_tid', 'Destination_tid']].astype(int).to_list()


def get_sorted_distances_from_tid(dist_mtx: pd.DataFrame,
                                  tid: int) -> pd.DataFrame:
    """_summary_

    Args:
        dist_mtx (_type_): _description_
        tid (_type_): _description_
    """
    df_list = []
    for col in ['Origin_tid', 'Destination_tid']:
        other_col = 'Destination_tid'
        if col == 'Destination_tid':
            other_col = 'Origin_tid'

        df = dist_mtx.loc[dist_mtx[col]==tid, [other_col, 'Total_Time']].copy()
        df.columns = ['TID', tid]
        df_list.append(df)

    result = pd.concat(df_list, ignore_index=True)
    result = result.groupby('TID').mean()
    return result.sort_values(by=tid).reset_index()


def get_route_from_outlier(dist_mtx: pd.DataFrame) -> List:
    """_summary_

    Args:
        dist_mtx (pd.DataFrame): _description_

    Returns:
        List: _description_
    """
    p1, _ = find_most_distant_points(dist_mtx)
    p1_dist = get_sorted_distances_from_tid(dist_mtx, p1)

    p1_tids = p1_dist['TID'].to_list()
    return [p1] + p1_tids


def clusterize_atm_dichotomous(loader: Data,
                               n_clusters: int,
                               allowed_percent=None,
                               tids_list: Optional[List[int]]=None,
                               random_state: Optional[int]=RS):
    """Дихотомическая кластеризация с учетом времени объезда банкоматов  в кластере

    Args:
        loader (Data): _description_
        n_clusters (int): _description_
        allowed_percent (_type_, optional): _description_. Defaults to None.
        tids_list (Optional[List[int]], optional): _description_. Defaults to None.
        random_state (Optional[int], optional): _description_. Defaults to RS.

    Returns:
        _type_: _description_
    """
    dist_mtx = loader.get_distance_matrix().copy()
    geo_df = loader.get_geo_TIDS().copy()

    if tids_list is not None:
        dist_mtx = dist_mtx[
            dist_mtx['Origin_tid'].isin(tids_list)
            &
            dist_mtx['Destination_tid'].isin(tids_list)
            ].copy()

        geo_df = geo_df[geo_df['TID'].isin(tids_list)].copy()

    _, _, sum_time = get_best_route(loader,
                                    geo_df['TID'].to_list(),
                                    n_iterations=1)

    mean_time = sum_time / n_clusters
    min_time = mean_time * 1.02
    max_time = mean_time * 1.05

    if allowed_percent is not None:
        max_time = min_time * (1 + 0.01 * allowed_percent)

    clusters = {}
    num_iter = 0

    while not dist_mtx.empty:

        clust = get_route_from_outlier(dist_mtx)
        print(f'Обработка кластера {num_iter:>3}, стартовый размер кластера: {len(clust)}')

        n = binary_search(clust, loader, min_time, max_time, n_iterations=1)

        clusters[num_iter] = clust[: n]
        cond = (
            dist_mtx['Origin_tid'].isin(clusters[num_iter])
            |
            dist_mtx['Destination_tid'].isin(clusters[num_iter])
            )

        dist_mtx = dist_mtx[~cond].copy()
        num_iter += 1

    geo_df[LABEL_COL] = 0

    if num_iter > n_clusters:
        warn(f'Решение не сошлось, получилось {num_iter} кластеров '
             f'(лишние точки будут добавлены в кластер {n_clusters-1}). '
             'Попробуйте увеличить allowed_percent')

        for j in range(n_clusters, num_iter):
            clusters[n_clusters-1] += clusters[j]
            clusters.pop(j, None)

    for i, tids in clusters.items():
        geo_df.loc[geo_df['TID'].isin(tids), LABEL_COL] = i

    return geo_df[['TID', LABEL_COL]].copy()


if __name__ == '__main__':
    my_loader = Data(RAW_DATA_FOLDER)
    print(clusterize_atm(my_loader,
                         n_clusters=10,
                         allowed_percent=0.01))
