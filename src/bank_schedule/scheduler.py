"""Скрипты для построения расписания инкассации на текущий день
"""

from typing import Dict, List, Optional
from warnings import warn
import pandas as pd
import numpy as np

import scipy.stats

from bank_schedule.data import Data
from bank_schedule.forecast import load_model, LGBM_MODEL_NAME

INCOME_COL = 'money_in'
RESIDUALS_COL = 'money'
DATE_COL = 'date'

# количество дней до переполнения банкомата
# или до 14 дней без обслуживания
DAYS_TO_DEADLINE_COL = 'days_to_deadline'

# дата прошлого посещения банкомата инкассаторами
LAST_COLLECTION_DATE_COL = 'last_collection_date'

# дата, начиная с которой нам известны данные об остатках
INITIAL_DATE = '2022-08-31'


def get_initial_resuduals(loader: Data,
                          initial_date: str=INITIAL_DATE,
                          last_collection_equal_to_initial: bool=True):
    """_summary_

    Args:
        loader (Data): _description_
        initial_date (str, optional): _description_. Defaults to INITIAL_DATE.
    """
    residuals = loader.get_money_start()
    incomes = loader.get_money_in()

    residuals[DATE_COL] = pd.to_datetime(initial_date)
    residuals = add_last_cash_collection_date(
        residuals, incomes, equal_to_initial=last_collection_equal_to_initial
        )

    return residuals


def prepare_residual_to_schedule_creation(residuals: pd.DataFrame,
                                          forecast_model: object,
                                          horizon: int) -> pd.DataFrame:
    """_summary_

    Args:
        residuals (pd.DataFrame): _description_
        forecast_model (object): _description_
        horizon (int): _description_

    Returns:
        _type_: _description_
    """
    residuals = add_nearest_overflow_date(residuals,
                                          forecast_model=forecast_model,
                                          horizon=horizon)

    residuals = calc_and_add_days_to_deadline_column(residuals)

    return residuals


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
    today = residuals[DATE_COL].unique()
    if today.shape[0] > 1:
        raise ValueError('More than one unique date in the residuals dataframe')
    return pd.to_datetime(today[0])


def calc_and_add_days_to_deadline_column(residuals: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку days_to_deadline

    Args:
        residuals (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    residuals = residuals.copy()
    days_to_collecting = 14 - (residuals[DATE_COL] - residuals['last_collection_date']).dt.days
    days_to_overflow_thresh = (residuals['overflow_date'] - residuals[DATE_COL]).dt.days

    residuals[DAYS_TO_DEADLINE_COL] = list(map(min, days_to_collecting, days_to_overflow_thresh))
    residuals[DAYS_TO_DEADLINE_COL] = residuals[DAYS_TO_DEADLINE_COL].astype(int)

    return residuals


def add_nearest_overflow_date(today_residuals: pd.DataFrame,
                              forecast_model: object,
                              horizon: int=3,
                              overflow_thresh: int=10**6) -> List[int]:
    """Проставляем ближайшие ожидаемые даты переполнения банкоматов, если оно произойдет
     в течение horizon дней
    1. Прогнозируем income на horizon дней вперед
    2. На основе прогноза определяем даты переполнения банкоматов
     в пределах заданного горизонта horizon
    3. Проставляем даты переполнения в рамках горизонта
    Предполагается, что мы делаем это вечером после инкассации

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
    predicted_dates = np.sort(in_preds[DATE_COL].unique())

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
        date_in_preds = in_preds[in_preds[DATE_COL] == date].copy()
        date_in_preds = date_in_preds.loc[today_residuals.index, :]

        # пополнение
        money += date_in_preds[INCOME_COL]

        overflow_cond = (money >= overflow_thresh) & today_residuals['overflow_date'].isna()

        today_residuals.loc[overflow_cond, 'overflow_date'] = date

    return today_residuals.reset_index()


def add_last_cash_collection_date(residuals: pd.DataFrame,
                                  income: pd.DataFrame,
                                  equal_to_initial: bool=True) -> pd.DataFrame:
    """Считает ориентировочную дату последней инкассации
    на момент 2022-08-31 исходя из средней заполняемости банкомата в день

    Args:
        residuals (pd.DataFrame): _description_
        income (pd.DataFrame): _description_
    """
    if equal_to_initial:
        residuals['last_collection_date'] = pd.to_datetime(INITIAL_DATE)
        return residuals

    avg_day_income = income.groupby('TID')[INCOME_COL].mean()

    residuals = residuals.set_index('TID').loc[avg_day_income.index, :]
    residuals_series = residuals[RESIDUALS_COL]

    days_from_last_collection = np.ceil(residuals_series / avg_day_income).astype(int)
    days_from_last_collection = days_from_last_collection.apply(lambda x: min(x, 14))

    residuals['last_collection_date'] = residuals[DATE_COL] - \
        pd.to_timedelta(days_from_last_collection, unit='D')

    return residuals.reset_index()


def calc_label_to_weights_map(data: pd.DataFrame,
                              label_column: str=DAYS_TO_DEADLINE_COL
                              ) -> Dict[str, int]:
    """Возвращает словарь с весами классов для выборки их из данных
    Веса нужны, чтобы выбирать сбалансированный случайный сэмпл

    Args:
        data (pd.DataFrame): данные с остатками в банкоматах на текущий день,
         спрогнозированными датами переполняемости на горизонте 14 дней и датами
         окончания срока без обслуживания
        label_column (str, optional): колонка с количеством дней до обязательной
         инкассации банкомата.
        Defaults to DAYS_TO_DEADLINE_COL.

    Returns:
        Dict[str, int]: _description_
    """

    mynnorm = scipy.stats.norm(0, 7)
    return {i: mynnorm.pdf(i) for i in range(100)}

    # n_samples_by_classes = data[label_column].value_counts()
    # n_samples = n_samples_by_classes.sum()
    # n_classes = n_samples_by_classes.shape[0]
    # classes_weights = n_samples / (n_classes * n_samples_by_classes)
    # classes_weights = classes_weights.to_dict()
    # return classes_weights


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


def get_weights_from_residuals(residuals: pd.DataFrame,
                               std: float=7.0) -> Dict:
    """Получает веса для банкоматов по их близости к дедлайну
    (по обслуживанию и переполнению)

    Args:
        residuals (pd.DataFrame): _description_

    Returns:
        Dict: _description_
    """
    weights_base = residuals[['TID', 'days_to_deadline']].drop_duplicates()
    mynnorm = scipy.stats.norm(0, std)
    weights_base['weight'] = weights_base['days_to_deadline'].apply(mynnorm.pdf)
    return weights_base.set_index('TID')['weight'].to_dict()


def get_non_mandatory_atms_samples(data: pd.DataFrame,
                                   n_samples: int,
                                   mandatory_selection_threshold: int,
                                   mandatory_selection_col: str=DAYS_TO_DEADLINE_COL,
                                   tids_list: Optional[List[int]]=None) -> pd.DataFrame:
    """Отбирает для инкассации на сегодня банкоматы, которые не обязательно
    обслуживать именно сегодня, но если обслужим - сбалансируем выборку

    Args:
        data (pd.DataFrame): _description_
        n_samples (int): _description_
        mandatory_selection_threshold (int): _description_
        mandatory_selection_col (str, optional): _description_. Defaults to DAYS_TO_DEADLINE_COL.
        tids_list (Optional[List[int]], optional): среди каких банкоматов выбирать.
         Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    non_mandatory = data[data[mandatory_selection_col] > mandatory_selection_threshold].copy()
    non_mandatory['weights'] = calc_samples_weights(non_mandatory, mandatory_selection_col)

    if tids_list is not None:
        non_mandatory = non_mandatory[non_mandatory['TID'].isin(tids_list)].copy()

    total_samples = non_mandatory.shape[0]

    if n_samples > total_samples:
        warn(f'В выборке {total_samples} точек. Это меньше, чем {n_samples}. '
             'Всё, что есть, будет отобрано.')
        n_samples = total_samples

    non_mandatory = non_mandatory.sample(n=n_samples,
                                         weights='weights',
                                         replace=False,
                                         random_state=0)

    return non_mandatory.drop(columns='weights')


def get_mandatory_samples(data: pd.DataFrame,
                          mandatory_selection_threshold: int,
                          mandatory_selection_col: str=DAYS_TO_DEADLINE_COL) -> pd.DataFrame:
    """Возвращает срез данных с банкоматами, которые обязательно обслужить сегодня
    (сегодня вечером будут переполнены или сегодня - 14 день без обслуживания

    Args:
        data (pd.DataFrame): _description_
        n_samples (int): _description_
        mandatory_selection_threshold (int): _description_
        mandatory_selection_col (str, optional): _description_. Defaults to DAYS_TO_DEADLINE_COL.
    """
    return data[data[mandatory_selection_col] <= mandatory_selection_threshold].copy()


def get_atms_for_today_collection(loader: Data,
                                  residuals: pd.DataFrame,
                                  n_samples: int,
                                  mandatory_selection_threshold: int,
                                  neighborhood_radius: 15,
                                  mandatory_selection_col: str=DAYS_TO_DEADLINE_COL,
                                  tids_col: str='TID',
                                  use_greedy: bool=True) -> pd.DataFrame:
    """Возвращает срез датафрейма остатков с банкоматами, которые нукжно сегодня инкассировать

    Args:
        residuals (pd.DataFrame): остатки в банкоматах на вчерашний вечер
        n_samples (int): сколько банкоматов мы должны сегодня инкассировать
        mandatory_selection_threshold (int): банкоматы с каким количеством дней до дедлайна
         мы должны обязательно инкассировать сегодня
        mandatory_selection_col (str, optional): в какой колонке лежит рассчитанное количество
         дней до дедлайна. Defaults to DAYS_TO_DEADLINE_COL.
        tids_col (str, optional): колонка с айди банкоматов. Defaults to 'TID'.

    Returns:
        pd.DataFrame: срез датафрейма остатков с банкоматами, которые нукжно сегодня инкассировать
    """
    mandatory = get_mandatory_samples(residuals, mandatory_selection_threshold,
                                      mandatory_selection_col)
    mandatory['is_mandatory'] = 1

    n_samples_non_mandatory = n_samples - mandatory.shape[0]

    if n_samples_non_mandatory <= 0:
        warn(
            f'Банкоматов, обязательных для обслуживания - {mandatory.shape[0]} (> {n_samples}). '
            'Будут инкассироваться только они, соседние не добавляем.'
            )

        if mandatory.duplicated(subset=tids_col).any():
            raise ValueError('Дублируются банкоматы в датафрейме остатков')

        return mandatory

    neighbours_list = get_neighbours(mandatory[tids_col].to_list(),
                                     loader,
                                     radius=neighborhood_radius)
    if use_greedy:
        non_mandatory = get_atms_for_today_collection_greedy(residuals,
                                                             n_samples_non_mandatory,
                                                             mandatory_selection_threshold,
                                                             tids_list=neighbours_list)
    else:
        non_mandatory = get_non_mandatory_atms_samples(residuals,
                                                       n_samples_non_mandatory,
                                                       mandatory_selection_threshold,
                                                       mandatory_selection_col,
                                                       tids_list=neighbours_list)

    non_mandatory['is_mandatory'] = 0

    atms_df = pd.concat([non_mandatory, mandatory], axis=0, ignore_index=True)

    if atms_df.duplicated(subset=tids_col).any():
        raise ValueError('Дублируются банкоматы в датафрейме остатков')

    return atms_df


def get_neighbours(tids_list: List[int],
                   loader: Data,
                   radius: float=15,
                   ) -> List[int]:
    """Получает соседние банкоматы для банкоматов из tids_list

    Args:
        tids_list List[int]: _description_
        loader (Data): _description_
        radius (float, optional): _description_. Defaults to 15.

    Returns:
        List[int]: _description_
    """
    tids_set = set(tids_list)

    distances_df = loader.get_distance_matrix()

    cond1 = distances_df['Origin_tid'].isin(tids_set)
    cond2 = distances_df['Destination_tid'].isin(tids_set)

    time_cond = distances_df['Total_Time'] < radius

    nb_set1 = set(distances_df.loc[cond1 & time_cond, 'Destination_tid'])
    nb_set2 = set(distances_df.loc[cond2 & time_cond, 'Origin_tid'])

    result = nb_set1.union(nb_set2)

    return list(result.difference(tids_set))

def get_atms_for_today_collection_greedy(residuals: pd.DataFrame,
                                         n_samples: int=200,
                                         mandatory_selection_threshold: int=1,
                                         tids_list: Optional[List[int]]=None
                                         ):
    """Жадно отбирает банкоматы для инкассации

    Args:
        residuals (pd.DataFrame): _description_
        n_samples (int, optional): _description_. Defaults to 200.
    """
    cond = [True for _ in range(residuals.shape[0])]

    if tids_list is not None:
        tids_list = set(tids_list)
        cond = residuals['TID'].isin(tids_list)

    counts_by_days = residuals.loc[cond, 'days_to_deadline'].value_counts().sort_index().to_dict()

    free_space = n_samples
    to_collect_list = []

    for i, cnt in counts_by_days.items():
        if free_space <=0:
            break

        to_collect_cnt = min(cnt, free_space)

        if i <= mandatory_selection_threshold:
            to_collect_cnt = cnt

        df = residuals[residuals['days_to_deadline']==i].sample(n=to_collect_cnt, random_state=0)
        df['is_mandatory'] = int(i <= mandatory_selection_threshold)

        to_collect_list.append(df.copy())
        free_space -= to_collect_cnt
    if free_space > 0:
        warn(f'В выборке {n_samples - free_space} точек. Это меньше, чем {n_samples}. '
             'Всё, что есть, будет отобрано.')
    return pd.concat(to_collect_list, ignore_index=True)
