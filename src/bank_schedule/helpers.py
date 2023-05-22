"""Вспомогательные функции
"""
import os
from tqdm import tqdm

from math import pi

import numpy as np
import pandas as pd

from bank_schedule.constants import (
    EARTH_R,
    CENTER_LAT,
    CENTER_LON,
    RAW_DATA_FOLDER,
    INTERIM_DATA_FOLDER
)

from bank_schedule.data import Data


def haversine_vectorized(
        data: pd.DataFrame,
        lat1_col: str,
        lon1_col: str,
        lat2_col: str,
        lon2_col: str
        ) -> pd.Series:
    """Считает расстояниие между точками,
    координаты которых заданы в виде широты и долготы
    по всему датафрейму

    Args:
        data (pd.DataFrame): датафрейм с геокоординатами точек
        latx_col (str): колонка с широтой точки 1
        lonx_col (str): колонка с долготой точки 1
        laty_col (str): колонка с широтой точки 2
        lony_col (str): колонка с долготой точки 2

    Returns:
        pd.Series: _description_
    """

    # convert decimal degrees to radians
    rcoords = {}

    for col in [lat1_col, lon1_col, lat2_col, lon2_col]:
        rcoords[col] = pi * data[col] / 180
    # haversine formula
    # The Haversine (or great circle) distance is the angular distance between
    #  two points on the surface of a sphere.
    #  The first coordinate of each point is assumed to be the latitude,
    #  the second is the longitude, given in radians
    dlon = abs(rcoords[lon2_col] - rcoords[lon1_col])
    dlat = abs(rcoords[lat2_col] - rcoords[lat1_col])

    hav_arg = np.sin(dlat/2)**2 + \
        np.cos(rcoords[lat1_col]) * np.cos(rcoords[lat2_col]) * (np.sin(dlon/2)**2)
    hav_dist = 2 * np.arcsin(np.sqrt(hav_arg))
    distance = EARTH_R * hav_dist
    return distance


def calc_cartesian_coords(lat_series: pd.Series,
                          lon_series: pd.Series,
                          center_lat: float=CENTER_LAT,
                          center_lon: float=CENTER_LON) -> pd.DataFrame:
    """Переводит геокоординаты в декартовы, считая нулем координат
     центр Москвы

    Args:
        lat_series (pd.Series): _description_
        lon_series (pd.Series): _description_
        moscow_center_lat (float): _description_
        moscow_center_lon (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cart_coords = pd.DataFrame()
    cart_coords['lat'] = np.asarray(lat_series)
    cart_coords['lon'] = np.asarray(lon_series)
    cart_coords['c_lat'] = center_lat
    cart_coords['c_lon'] = center_lon
    cart_coords['x'] = haversine_vectorized(cart_coords, 'c_lat', 'c_lon', 'c_lat', 'lon')
    cart_coords['y'] = haversine_vectorized(cart_coords, 'c_lat', 'c_lon', 'lat', 'c_lon')
    minus_cond_x = cart_coords['c_lon'] > cart_coords['lon']
    minus_cond_y = cart_coords['c_lat'] > cart_coords['lat']
    cart_coords.loc[minus_cond_x, 'x'] = cart_coords.loc[minus_cond_x, 'x'] * (-1)
    cart_coords.loc[minus_cond_y, 'y'] = cart_coords.loc[minus_cond_y, 'y'] * (-1)

    return cart_coords[['x', 'y']]


def calculate_tid_income_cross_correlations() -> pd.DataFrame:
    """Считает кросс-корреляции между наполнением различных TID
    """
    loader = Data(RAW_DATA_FOLDER)
    colnames = ['TID', 'other_TID', 'pearson_coef']

    in_df = loader.get_money_in()

    all_tid = in_df['TID'].unique()
    all_crosscor_tid = []

    for i, tid in enumerate(tqdm(all_tid)):
        data_tid = in_df.loc[in_df['TID'] == tid, 'money_in'].values
        other_tid = all_tid[i+1:]

        for other in other_tid:
            data_other_tid = in_df.loc[in_df['TID'] == other, 'money_in'].values
            corr = np.corrcoef(data_tid, data_other_tid)[0, 1]
            all_crosscor_tid.append([tid, other_tid, corr])

    crosscor_tid_df = pd.DataFrame(all_crosscor_tid)
    crosscor_tid_df.columns = colnames
    crosscor_tid_df['pearson_coef'] = crosscor_tid_df['pearson_coef'].round(3)
    crosscor_tid_df['abs_pearson_coef'] = crosscor_tid_df['pearson_coef'].abs()

    crosscor_tid_df = crosscor_tid_df.sort_values(by='abs_pearson_coef',
                                                  ascending=False)

    return crosscor_tid_df[colnames] # pylint: disable=E1136


def get_tid_income_cross_correlations(recalculate: bool=False):
    """Считывает или рассчитывает, сохраняет и считывает
    файл с кросс-корреляциями пополнений для различных TID

    Args:
        recalculate (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    filepath = os.path.join(INTERIM_DATA_FOLDER, 'tid_income_cross_correlations.csv')
    if recalculate or not os.path.isfile(filepath):
        crosscor_tid_df = calculate_tid_income_cross_correlations()
        crosscor_tid_df.to_csv(filepath, index=False)
    return pd.read_csv(filepath)


def extract_features_from_date(date_series: pd.Series) -> pd.DataFrame:
    """Extract day, month, week, weekday from date_series.

    Args:
        date_series (pd.Series): _description_

    Returns:
        pd.DataFrame: _description_
    """
    features = pd.DataFrame()
    features["day"] = date_series.dt.day
    features["weekday"] = date_series.dt.weekday
    return features


def create_targets(in_df: pd.DataFrame,
                   max_period: int=30) -> pd.DataFrame:
    """_summary_

    Args:
        in_df (pd.DataFrame): _description_
        max_period (int, optional): _description_. Defaults to 30.

    Returns:
        pd.DataFrame: _description_
    """

    all_tids = in_df['TID'].unique().tolist()
    tgt_df_list = []

    for tid in tqdm(all_tids):
        tgt_df = in_df.loc[in_df['TID'] == tid, ['date', 'TID', 'money_in']].copy()

        for shift in range(1, max_period+1):
            tgt_df[f'target_income_{shift}_day_after'] =  tgt_df['money_in'].shift(-shift)

        tgt_df_list.append(tgt_df)
    all_tgt_df = pd.concat(tgt_df_list, ignore_index=True)
    all_tgt_df.drop(columns=['money_in'], inplace=True)
    return all_tgt_df


def create_features(in_df: pd.DataFrame,
                    max_period: int=30):
    """Создает фичи

    Args:
        in_df (pd.DataFrame): _description_
        max_period (int, optional): _description_. Defaults to 30.
    """
    all_tids = in_df['TID'].unique().tolist()
    diff_df_list = []

    for tid in tqdm(all_tids):
        diff_df = in_df.loc[in_df['TID'] == tid, ['date', 'TID', 'money_in']].copy()
        diff_df.loc[:, 'yesterday_income_growth'] = diff_df['money_in'].diff(1).shift()
        diff_df.loc[:, 'yesterday_growth_of_income_growth'] = \
            diff_df['yesterday_income_growth'].diff(1).shift()

        for shift in range(1, max_period+1):
            diff_df[f'income_{shift}_day_before'] = diff_df['money_in'].shift(shift)

        diff_df['mean_income_all'] = diff_df['money_in'].mean()
        diff_df['mean_income_last_30_days'] = diff_df['money_in'].rolling(window=30).mean().shift(1)
        diff_df['mean_income_last_7_days'] = diff_df['money_in'].rolling(window=7).mean().shift(1)
        diff_df['mean_income_last_3_days'] = diff_df['money_in'].rolling(window=3).mean().shift(1)

        diff_df_list.append(diff_df)

    all_diff_df = pd.concat(diff_df_list, ignore_index=True)
    all_diff_df.drop(columns=['money_in'], inplace=True)

    return all_diff_df

if __name__ == '__main__':
    get_tid_income_cross_correlations(recalculate=False)
