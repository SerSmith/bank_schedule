"""Скрипты прогноза временного ряда
"""
import os
from typing import Dict, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb

from bank_schedule import helpers
from bank_schedule.constants import LGBM_FOLDER


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


    def predict(self, n_periods: int) -> pd.Series:
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


def train_lgbm_models(max_forecast_horizon: int=30,
                      private_days_count: int=30,
                      n_iterations: int=5,
                      model_parameters: Optional[Dict]=None) -> Dict:
    """_summary_

    Args:
        max_forecast_horizon (int, optional): _description_. Defaults to 30.
        private_days_count (int, optional): _description_. Defaults to 30.
        n_iterations (int, optional): _description_. Defaults to 5.
        model_parameters (Optional[Dict], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Dict: _description_
    """
    if model_parameters is None:
        model_parameters = dict(
            n_estimators=1000,
            num_leaves=31,
            bagging_fraction=.7,
            feature_fraction=.7,
            random_state=27,
            early_stopping_round=5
    )

    # загружаем трейн
    train_data = helpers.read_or_create_train_data(recalculate=False)

    # разделяем его по дням года на публичную и приватную часть
    dayofyear = helpers.extract_dayofyear_series(train_data)

    unique_dayofyear = dayofyear.unique()
    unique_dayofyear.sort()

    public_days = unique_dayofyear[:-private_days_count]

    # колонки с признаками
    features_cols = [col for col in train_data.columns if 'target' not in col]

    # колонка с true target (её значение берем для валидации за нужный день)
    true_target = 'money_in'

    for forecast_horizon in tqdm(range(1, max_forecast_horizon + 1)):
        train_days = public_days[:-forecast_horizon]
        val_days = [public_days[-1]]

        if val_days[0] - train_days[-1] != forecast_horizon:
            raise ValueError('Ошибка в расчете дней для обучения')

        # на какой таргет учимся
        needed_target = f'target_income_{forecast_horizon}_day_after'

        # обучаемся с валидацией и находим лучшее количество решателей
        best_iterations = []

        for i in range( - n_iterations + 1, 1 ):
            shift = i
            if i == 0:
                shift = None

            x_train = helpers.filter_train_by_needed_days(train_data, train_days[:shift])
            x_val = helpers.filter_train_by_needed_days(train_data, [val_days[0]-i])

            # разделяем на признаки и таргеты
            x_train, y_train = x_train[features_cols], x_train[needed_target]
            x_val, y_val = x_val[features_cols], x_val[true_target]

            model = lgb.LGBMRegressor(**model_parameters)

            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                eval_metric='mae'
            )
            best_iterations.append(model.best_iteration_)

        new_params = {k: v for k, v in model_parameters.items()
                      if k not in {'early_stopping_round', 'n_estimators'}}

        new_params['early_stopping_round'] = None
        new_params['n_estimators'] = int(np.mean(best_iterations))

        # расширяем трейн и обучаемся на всех данных с подобранным числом
        # решателей
        extended_train_days = list(range(train_days[0],
                                        val_days[-1] + 1,
                                        1))

        if extended_train_days != list(public_days):
            print(extended_train_days, public_days)
            raise ValueError('Ошибка в расчете дней для обучения')

        # собираем полный трейн и обучаем модель
        new_train = helpers.filter_train_by_needed_days(train_data,
                                                        extended_train_days)

        new_x_train, new_y_train = new_train[features_cols], new_train[needed_target]
        model = lgb.LGBMRegressor(**new_params)

        model.fit(new_x_train, new_y_train)

        path = os.path.join(LGBM_FOLDER, f'{forecast_horizon}.txt')
        model.booster_.save_model(path)


class IncomeForecastLGBM():
    """Прогноз income бустингом
    """
    def __init__(self) -> None:
        self.models = self.__load_models()
        if not self.models:
            raise ValueError('Модели не найдены, запустите forecast.train_lgbm_models()')

        self.train = helpers.read_or_create_train_data(recalculate=False)
        features = [col for col in self.train.columns if 'target' not in col]
        self.train = self.train[features]


    def __load_models(self):
        models = {}
        for file in os.listdir(LGBM_FOLDER):
            if file.endswith('.txt'):
                model_name = int(file.split('.')[0])
                models[model_name] = lgb.Booster(
                    model_file=os.path.join(LGBM_FOLDER, file)
                )
        return models


    def predict(self,
                today_date: Union[str, np.datetime64],
                n_periods: int) -> pd.DataFrame:
        """_summary_

        Args:
            today_date (str | np.datetime64): "сегодняшняя" дата
            n_periods (int): на сколько дней вперед делать прогноз

        Returns:
            pd.DataFrame: _description_
        """
        today_date = pd.to_datetime(today_date)
        cond = self.train.index == today_date
        if cond.sum() == 0:
            raise ValueError(f'Дата {today_date} не найдена в данных')

        slice_df = self.train[cond]
        predictions_list = []

        for i in range(1, n_periods + 1):

            if i not in self.models:
                warn('Горизонт прогноза слишком большой')
                pred_vals = []

                for _, mdl in self.models.items():
                    pred_vals.append(mdl.predict(slice_df))

                pred_vals = np.concatenate(
                    pred_vals
                    ).reshape(len(self.models), slice_df.shape[0])
                pred_vals = pred_vals.mean(axis=0)

            else:
                pred_vals = self.models[i].predict(slice_df)

            predictions = pd.DataFrame(pred_vals, columns=['money_in'])
            predictions['TID'] = slice_df['TID'].values
            predictions['date'] = today_date + pd.Timedelta(days=i)
            predictions_list.append(predictions)

        return pd.concat(predictions_list)[['date', 'TID', 'money_in']]


if __name__=='__main__':

    # train_lgbm_models()
    # print('train_lgbm_models done')

    my_model = IncomeForecastLGBM()
    print(my_model.predict(pd.to_datetime('2022-11-30'), 30))
