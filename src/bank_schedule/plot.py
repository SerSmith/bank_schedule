"""Скрипты для визуализации
"""
import os
from typing import Optional, Union, List
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

import plotly.express as px

from bank_schedule.helpers import calc_cartesian_coords
from bank_schedule.constants import CENTER_LAT, CENTER_LON, EXTERNAL_DATA_FOLDER

def plot_clusters_size(labels_series: pd.Series,
                       title: str='') -> None:
    """_summary_

    Args:
        labels_series (pd.Series): _description_
    """
    labels_count = labels_series.value_counts() / labels_series.shape[0]
    plt.bar(x=labels_count.index, height=labels_count)
    if title:
        plt.title(title)
    plt.show()


def geoplot_clusters(coords_df: pd.DataFrame,
                     labels_col: str,
                     lat: str='latitude',
                     lon: str='longitude',
                     discrete_colors: bool=True,
                     html_folder: str=''):
    """_summary_

    Args:
        coords_df (pd.DataFrame): _description_
        labels_col (str): _description_
        lat (str, optional): _description_. Defaults to 'latitude'.
        lon (str, optional): _description_. Defaults to 'longitude'.
        discrete_colors (bool, optional): _description_. Defaults to True.
        html_folder (str, optional): _description_. Defaults to ''.
    """

    if discrete_colors:
        coords_df[labels_col] = coords_df[labels_col].astype(str)

    fig = px.scatter_mapbox(coords_df,
                            lat=lat,
                            lon=lon,
                            color=labels_col,
                            center={'lat': CENTER_LAT, 'lon': CENTER_LON},
                            height=800,
                            width=800,
                            opacity=.9,
                            mapbox_style='carto-positron',
                            title='Кластеры банкоматов')

    fig.write_html(os.path.join(html_folder, 'clusters.html'))
    fig.show()


def plot_normal_distr(mean: float, std: float):
    """Выводит график функции плотности для нормального рапределения

    Args:
        mean (_type_): _description_
        std (_type_): _description_
    """
    mynorm = norm(mean, std)
    x_vals = list( np.arange(-3 * std, 3 * std + 0.1, .1) )
    y_vals = pd.Series(data=[mynorm.pdf(x) for x in x_vals], index=x_vals)
    plt.plot(y_vals)


def plot_map(
    cartes1: pd.DataFrame,
    cartes2: Optional[pd.DataFrame]=None,
    size1: int=10,
    size2: int=1,
    alpha1: float=.2,
    alpha2: float=1,
    c1: Union[str, List]='b',
    c2: Union[str, List]='r'):
    """Печатает точки на фоне предсохраненной карты Москвы

    Args:
        cartes1 (pd.DataFrame): _description_
        cartes2 (_type_, optional): _description_. Defaults to None.
        size1 (int, optional): _description_. Defaults to 10.
        size2 (int, optional): _description_. Defaults to 1.
        alpha1 (float, optional): _description_. Defaults to .2.
        alpha2 (int, optional): _description_. Defaults to 1.
        c1 (str, optional): _description_. Defaults to 'b'.
        c2 (str, optional): _description_. Defaults to 'r'.
    """

    mos_img = plt.imread(os.path.join(EXTERNAL_DATA_FOLDER, 'map.png'))

    bbox_geo = (37.3260, 37.9193, 55.5698, 55.9119)
    bbox_cartes = calc_cartesian_coords(bbox_geo[2:], bbox_geo[:2])
    bbox = bbox_cartes['x'].to_list() + bbox_cartes['y'].to_list()

    _, ax = plt.subplots(figsize=(12,12))
    ax.scatter(cartes1['x'], cartes1['y'], zorder=1, alpha=alpha1, c=c1, s=size1)
    if cartes2 is not None:
        ax.scatter(cartes2['x'], cartes2['y'], zorder=1, alpha=alpha2, c=c2, s=size2)

    ax.set_xlim(bbox[0],bbox[1])
    ax.set_ylim(bbox[2],bbox[3])
    ax.axis('off')
    ax.imshow(mos_img, zorder=0, extent=bbox, aspect='equal')
    plt.show()
