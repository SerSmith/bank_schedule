"""Скрипты прогноза временного ряда
"""
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
    )

def classic_model_ts_cv(model,
                        features,
                        targets,
                        metric: str='rmse',
                        n_splits: int=5) -> List:
    """Кросс-валидация классической модели (бустинг etc.) для прогноза временных рядом

    Args:
        model (_type_): _description_
        features (_type_): _description_
        targets (_type_): _description_
        metric (str, optional): _description_. Defaults to 'rmse'.
        n_splits (int, optional): _description_. Defaults to 5.

    Returns:
        List: _description_
    """

    score_list = []
    cross_validator = TimeSeriesSplit(n_splits=n_splits)

    for train_index, valid_index in cross_validator.split(targets):

        train_index = targets.index[train_index]
        valid_index = targets.index[valid_index]

        features_train, features_valid = features.loc[train_index,:], features.loc[valid_index,:]

        targets_train, targets_valid = targets[train_index], targets[valid_index]

        model.fit(features_train, targets_train)

        predictions = model.predict(features_valid)

        if metric == 'mae':
            score = mean_absolute_error(targets_valid, predictions)
        elif metric == 'rmse':
            score = np.sqrt(mean_squared_error(targets_valid, predictions))
        elif metric == 'r2':
            score = r2_score(targets_valid, predictions)

        score_list.append(score)

    return score_list


def arima_model_ts_cv(model,
                      targets,
                      metric: str='rmse',
                      n_splits: int=5) -> List:
    """Кросс-валидация классической модели (бустинг etc.) для прогноза временных рядом

    Args:
        model (_type_): _description_
        features (_type_): _description_
        targets (_type_): _description_
        metric (str, optional): _description_. Defaults to 'rmse'.
        n_splits (int, optional): _description_. Defaults to 5.

    Returns:
        List: _description_
    """

    score_list = []
    cross_validator = TimeSeriesSplit(n_splits=n_splits)

    for train_index, valid_index in cross_validator.split(targets):

        train_index = targets.index[train_index]
        valid_index = targets.index[valid_index]

        targets_train, targets_valid = targets[train_index], targets[valid_index]

        model.fit(targets_train)

        predictions = model.predict(n_periods=len(targets_valid))

        if metric == 'mae':
            score = mean_absolute_error(targets_valid, predictions)
        elif metric == 'rmse':
            score = np.sqrt(mean_squared_error(targets_valid, predictions))
        elif metric == 'r2':
            score = r2_score(targets_valid, predictions)

        score_list.append(score)

    return score_list


class SimpleRollingForecast:
    """class for simple timeseries forecast with rolling window and exponential smoothing
    """
    def __init__(self,
                 window,
                 alpha=0,
                 ):
        """_summary_

        Args:
            window (_type_): окно, по которому считается среднее
            alpha (float, optional): фактор сглаживания. Defaults to 0.
        """
        self.window = window
        self.alpha = alpha
        self.rolling_mean = None

    def fit(self, series: pd.Series):
        """считает среднее по скользящему окну с экспоненциальным сглаживанием

        Args:
            series (pd.Series): _description_
        """
        window_series = pd.Series(series).iloc[-self.window:]
        if self.alpha == 0:
            self.rolling_mean = window_series.mean()
            return
        # add exponential smoothing to window_series
        self.rolling_mean = window_series.ewm(alpha=self.alpha).mean().iloc[-1]


    def predict(self, n_periods: int):
        """прогнозирует просто средним по скользящему окну
        на заданное количество дней вперед

        Args:
            n_periods (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.rolling_mean is None:
            raise ValueError("Model not fitted")
        return pd.Series(self.rolling_mean * np.ones(n_periods))


if __name__=='__main__':
    from bank_schedule.data import Data
    from bank_schedule.constants import RAW_DATA_FOLDER

    loader = Data(data_folder=RAW_DATA_FOLDER)
    in_df = loader.get_money_in()

    tid_list = in_df['TID'].sample(n=5).to_list()

    # пример для нескольких TID и нескольких периодов
    for tid in tid_list:
        print(f'\nTID: {tid}')
        tid_cond = in_df['TID'] == tid

        # пример для нескольких разных периодов
        for n_periods in [1, 2, 5, 10, 30]:
            # Инициализация и обучение простой модели со скользящим средним
            # инициализируем модель
            model = SimpleRollingForecast(window=10, alpha=0)

            # разделяем трейн и тест
            train_tgts = in_df.loc[tid_cond, 'money_in'].values[:-n_periods]
            test_tgts = in_df.loc[tid_cond, 'money_in'].values[-n_periods:]

            # обучаем модель
            model.fit(train_tgts)

            # делаем прогноз
            predictions = model.predict(n_periods=n_periods)

            # считаем и выводим скор
            mae = mean_absolute_error(test_tgts, predictions)
            mae = round(mae, 2)

            non_zero_cond = test_tgts != 0
            mape = mean_absolute_percentage_error(test_tgts[non_zero_cond],
                                                predictions[non_zero_cond])
            mape = round(mape, 3)

            print(f'{n_periods} дней, MAE:, {mae}')
            print(f'{n_periods} дней, MAPE:, {mape}')
