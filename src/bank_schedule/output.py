from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

def find_ATM_balance(df_opt_result, df_money, df_money_start, params_dict):
    """
    Функция принимает результаты оптимизации и выдает датафрейм с остатками на счетах в банкоматах на каждый день
    """
    df_money = df_money.copy()
    df_money_start = df_money_start.set_index('TID').copy()
    df_money = df_money.set_index('TID')
    df_money.drop(['остаток на 31.08.2022 (входящий)'], axis=1, inplace=True)
    df_money.columns = pd.to_datetime(df_money.columns)

    df_money_sum = df_money.copy()

    for i in tqdm(df_money.columns):
        df_money_sum[i] = df_money[i] + df_money_start['money']
    for (j, column) in enumerate(df_money.columns):
        for column_2 in df_money.columns[j+1:]:
            df_money_sum[column_2] = df_money_sum[column_2] + df_money[column]

    date_start = df_money_sum.columns.min()
    date_minus_1  = date_start - timedelta(days=1)
    df_money_sum[date_minus_1] = df_money_start['money']
    df_money_sum = df_money_sum[df_money_sum.columns.sort_values()]


    unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
    dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
    dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}

    # обработаем инкасации
    for day in tqdm(unique_date):
        date_before = dict_num_to_date[dict_date_to_num[day]-1]
        for j in range(dict_date_to_num[day], df_money_sum.shape[1]):
            df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), dict_num_to_date[j]] -= df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), date_before]

    df_money_sum.drop([date_minus_1], axis=1, inplace=True)

    # ограничиваем максимальной вместительностью
    condition = df_money_sum < params_dict['max_money']
    df_money_sum = df_money_sum.where(condition, other = params_dict['max_money'])
    return df_money_sum

def find_ATM_balance2(df_opt_result, df_money_in, df_money_start, data):
    """
    Функция принимает результаты оптимизации и выдает датафрейм с остатками на счетах в банкоматах на каждый день
    """
    params_dict = data.get_params_dict()
    df_money = pd.DataFrame(index=df_money_in['TID'].unique())
    for date in df_money_in.date.unique():
        df_money[date] = df_money_in[df_money_in.date==date].set_index('TID')['money_in']
    df_money_start = df_money_start.set_index('TID').copy()
    df_money.columns = pd.to_datetime(df_money.columns)

    df_money_sum = df_money.copy()

    for i in tqdm(df_money.columns):
        df_money_sum[i] = df_money[i] + df_money_start['money']
    for (j, column) in enumerate(df_money.columns):
        for column_2 in df_money.columns[j+1:]:
            df_money_sum[column_2] = df_money_sum[column_2] + df_money[column]

    date_start = df_money_sum.columns.min()
    date_minus_1  = date_start - timedelta(days=1)
    df_money_sum[date_minus_1] = df_money_start['money']
    df_money_sum = df_money_sum[df_money_sum.columns.sort_values()]


    unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
    dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
    dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}

    # обработаем инкасации
    list_date = df_opt_result.date.unique()
    list_date.sort()
    for day in tqdm(list_date):
        day = pd.to_datetime(day)
        date_before = dict_num_to_date[dict_date_to_num[day]-1]
        for j in range(dict_date_to_num[day], df_money_sum.shape[1]):
            df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), dict_num_to_date[j]] -= df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), date_before]

    df_money_sum.drop([date_minus_1], axis=1, inplace=True)

    # ограничиваем максимальной вместительностью
    condition = df_money_sum < params_dict['max_money']
    df_money_sum = df_money_sum.where(condition, other = params_dict['max_money'])
    return df_money_sum


def find_cost_inc(df_money_sum, df_opt_result, df_money_start, params_dict):
    """
    Функция принимающая на вход остатки в банкоматах и результат оптимизации 
    и выдающая стоимость инкасаций, которые мы нашли в результате оптимизаций
    """
    date_start = df_money_sum.columns.min()
    date_minus_1  = date_start - timedelta(days=1)
    df_money_sum[date_minus_1] = df_money_start['money']
    df_money_sum = df_money_sum[df_money_sum.columns.sort_values()]


    unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
    dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
    dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}
    num_TID = df_money_sum.shape[0]

    df_cost_inc = pd.DataFrame(index=df_money_sum.index)
    for day in tqdm(unique_date):
        date_before = dict_num_to_date[dict_date_to_num[day]-1]
        #df_cost_inc[day] = [0] * num_TID
        df_cost_inc.loc[list(df_opt_result[df_opt_result.date==day]['TID']), day] = df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), date_before] * 0.00001


    df_cost_inc = df_cost_inc.fillna(0)

    condition = (df_cost_inc >= params_dict['cost_inc_min']) | (df_cost_inc == 0)
    df_cost_inc = df_cost_inc.where(condition, other = params_dict['cost_inc_min'])
    return df_cost_inc

def find_sum_fond(df_money_sum, params_dict):
    df_fond = df_money_sum * params_dict['overnight'] / 100 / 365
    return df_fond

def find_all_cost(df_money_sum, df_opt_result, df_money_start,data ):
    """
    """
    params_dict = data.get_params_dict()
    df_cost_inc = find_cost_inc(df_money_sum, df_opt_result, df_money_start, params_dict)
    df_fond = find_sum_fond(df_money_sum, params_dict)
    cost_auto = params_dict['price_car'] * len(df_opt_result.auto.unique()) * ((df_opt_result.date.max() - df_opt_result.date.min() ).days + 1 ) # или по датам лучше переписать на df_money_in ?
    all_cost = sum(df_fond.add(df_cost_inc, fill_value=0).sum()) + cost_auto
    return all_cost