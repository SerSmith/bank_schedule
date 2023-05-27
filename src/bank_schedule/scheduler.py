"""Скрипты для построения расписания инкассации на текущий день
"""

from typing import Dict, List
import pandas as pd
import numpy as np

INCOME_COL = 'money_in'
RESIDUALS_COL = 'money'


def get_today_from_residuals(residuals: pd.DataFrame) -> pd.Timestamp:
    """Из датафреймов с остатками в банкоматах извлекает дату,
    за вечер которой известны эти остатки

    Args:
        residuals (pd.DataFrame): _description_

    Raises:
        ValueError: _description_

    Returns:
        pd.Timestamp: _description_
    """
    today = residuals['date'].unique()
    if today.shape[0] > 1:
        raise ValueError('More than one unique date in the residuals dataframe')
    return pd.to_datetime(today[0])


def add_overflow_date(today_residuals: pd.DataFrame,
                      forecast_model: object,
                      horizon: int=3,
                      overflow_thresh: int=10**6) -> List[int]:
    """Проставляем даты переполнения банкоматов, если оно произойдет
     в течение horizon дней
    1. Прогнозируем income на horizon дней вперед
    2. На основе прогноза определяем даты переполнения банкоматов
     в пределах заданного горизонта horizon
    3. Проставляем даты переполнения в рамках горизонта

    Args:
        today_residuals (pd.DataFrame): данные об остатках на текущий вечер
        forecast_model (object): модель прогноза income
        horizon (int, optional): горизонт анализа. Defaults to 3.
        overflow_thresh (int, optional): порог переполнения. Defaults to 10**6.

    Raises:
        ValueError: _description_

    Returns:
        List[int]: _description_
    """
    # готовим данные об остатках денег в банкоматах на вечер текущего дня
    today_residuals = today_residuals.copy()
    today_residuals.set_index('TID', inplace=True)

    # получаем дату текущего дня
    today = get_today_from_residuals(today_residuals)

    # прогнозируем на следующие horizon дней вперед
    in_preds = forecast_model.predict(today, n_periods=horizon, income_threshold=None)
    in_preds.set_index('TID', inplace=True)

    # извлекаем даты, на которые сделали прогноз
    predicted_dates = np.sort(in_preds['date'].unique())

    # даты предполагаемых переполнений банкоматов
    today_residuals['overflow_date'] = pd.NaT

    # сумма, которая будет внесена за horizon дней
    # так ак нам надо вернуть исходный датафрейм с маркерами переполнения
    # то создадим отдельные series для этого
    money = today_residuals[RESIDUALS_COL].copy()

    # если уже переполнен, ставим текущую дату
    today_residuals.loc[money >= overflow_thresh, 'overflow_date'] = today

    for date in predicted_dates:

        if today_residuals['overflow_date'].isna().sum() == 0:
            break

        # прогнозы пополнений за текущую дату
        date_in_preds = in_preds[in_preds['date'] == date].copy()
        date_in_preds = date_in_preds.loc[today_residuals.index, :]

        # пополнение
        money += date_in_preds[INCOME_COL]

        overflow_cond = (money >= overflow_thresh) & today_residuals['overflow_date'].isna()

        today_residuals.loc[overflow_cond, 'overflow_date'] = date

    return today_residuals.reset_index()


def add_last_cash_collection_date(residuals: pd.DataFrame,
                                  income: pd.DataFrame,
                                  fake: bool=True) -> pd.DataFrame:
    """Считает ориентировочную дату последней инкассации
    на момент 2022-08-31 исходя из средней заполняемости банкомата в день

    Args:
        residuals (pd.DataFrame): _description_
        income (pd.DataFrame): _description_
    """
    if fake:
        residuals['last_collection_date'] = pd.to_datetime('2022-08-31')
        return residuals

    avg_day_income = income.groupby('TID')[INCOME_COL].mean()

    residuals = residuals.set_index('TID').loc[avg_day_income.index, :]
    residuals_series = residuals[RESIDUALS_COL]

    days_from_last_collection = np.ceil(residuals_series / avg_day_income).astype(int)
    days_from_last_collection = days_from_last_collection.apply(lambda x: min(x, 14))

    residuals['last_collection_date'] = residuals['date'] - \
        pd.to_timedelta(days_from_last_collection, unit='D')

    return residuals.reset_index()


def calc_label_to_weights_map(data: pd.DataFrame,
                              label_column: str='days_to_deadline'
                              ) -> Dict[str, int]:
    """Возвращает словарь с весами классов для выборки их из данных
    Веса нужны, чтобы выбирать сбалансированный случайный сэмпл

    Args:
        data (pd.DataFrame): данные с остатками в банкоматах на текущий день,
         спрогнозированными датами переполняемости на горизонте 14 дней и датами
         окончания срока без обслуживания
        label_column (str, optional): колонка с количеством дней до обязательной
         инкассации банкомата.
        Defaults to 'days_to_deadline'.

    Returns:
        Dict[str, int]: _description_
    """

    n_samples_by_classes = data[label_column].value_counts()
    n_samples = n_samples_by_classes.sum()
    n_classes = n_samples_by_classes.shape[0]
    classes_weights = n_samples / (n_classes * n_samples_by_classes)
    classes_weights = classes_weights.to_dict()
    return classes_weights


def calc_samples_weights(data: pd.DataFrame,
                         label_column: str) -> List[int]:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        label_column (str): _description_

    Returns:
        List[int]: _description_
    """
    base_to_weights_map = calc_label_to_weights_map(data, label_column)
    return data[label_column].map(base_to_weights_map).to_list()


def get_non_mandatory_atms_samples(data: pd.DataFrame,
                                   n_samples: int,
                                   mandatory_selection_threshold: int,
                                   mandatory_selection_col: str='days_to_deadline') -> pd.DataFrame:
    """Отбирает для инкассации на сегодня банкоматы, которые не обязательно
    обслуживать именно сегодня, но если обслужим - сбалансируем выборку

    Args:
        data (pd.DataFrame): _description_
        n_samples (int): _description_
        mandatory_selection_threshold (int): _description_
        mandatory_selection_col (str, optional): _description_. Defaults to 'days_to_deadline'.
        tids_col (str, optional): _description_. Defaults to 'TID'.

    Returns:
        pd.DataFrame: _description_
    """
    non_mandatory = data[data[mandatory_selection_col] > mandatory_selection_threshold].copy()
    weights = calc_samples_weights(non_mandatory, mandatory_selection_col)
    return non_mandatory.sample(n=n_samples, weights=weights)


def get_mandatory_samples(data: pd.DataFrame,
                          mandatory_selection_threshold: int,
                          mandatory_selection_col: str='days_to_deadline') -> pd.DataFrame:
    """Возвращает срез данных с банкоматами, которые обязательно обслужить сегодня
    (сегодня вечером будут переполнены или сегодня - 14 день без обслуживания

    Args:
        data (pd.DataFrame): _description_
        n_samples (int): _description_
        mandatory_selection_threshold (int): _description_
        mandatory_selection_col (str, optional): _description_. Defaults to 'days_to_deadline'.
    """
    return data[data[mandatory_selection_col] <= mandatory_selection_threshold].copy()


def get_atms_for_today_collection(data: pd.DataFrame,
                                  n_samples: int,
                                  mandatory_selection_threshold: int,
                                  mandatory_selection_col: str='days_to_deadline',
                                  tids_col: str='TID') -> pd.DataFrame:
    """Возвращает срез даннки с банкоматама, коллежирующимим для обслуживания
    текущимиддельникоматама

    Args:
        data (pd.DataFrame): _description_
        n_samples (int): _description_
        mandatory_selection_threshold (int): _description_
        mandatory_selection_col (str, optional): _description_. Defaults to 'days_to_deadline'.
        tids_col (str, optional): _description_. Defaults to 'TID'.

    Returns:
        pd.DataFrame: _description_
    """
    mandatory = get_mandatory_samples(data, mandatory_selection_threshold,
                                      mandatory_selection_col)
    n_samples_non_mandatory = n_samples - mandatory.shape[0]
    non_mandatory = get_non_mandatory_atms_samples(data,
                                                   n_samples_non_mandatory,
                                                   mandatory_selection_threshold,
                                                   mandatory_selection_col)
    atms_df = pd.concat([non_mandatory, mandatory], axis=0, ignore_index=True)

    if atms_df.duplicated(subset=tids_col).any():
        raise ValueError('Дублируются банкоматы в датафрейме остатков')

    return atms_df
