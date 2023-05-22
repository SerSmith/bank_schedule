"""Скрипты для визуализации
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from bank_schedule.constants import CENTER_LAT, CENTER_LON

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