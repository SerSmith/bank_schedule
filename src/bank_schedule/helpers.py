"""Вспомогательные функции
"""

from math import pi

import numpy as np
import pandas as pd

from bank_schedule.constants import (
    EARTH_R,
    CENTER_LAT,
    CENTER_LON
)

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
