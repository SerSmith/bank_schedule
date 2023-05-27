from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def condition_max_days_inc(df_opt_result, df_money_in, data):
    params_dict = data.get_params_dict()
    unique_tid = list(df_opt_result.TID.unique())
    date_last = df_money_in.date.max()
    date_first = df_money_in.date.min()
    for tid in tqdm(unique_tid):
        df_opt_result_tid = df_opt_result[df_opt_result.TID==tid].copy().sort_values('date').reset_index(drop=True)
        df_opt_result_tid.loc[df_opt_result_tid.shape[0]] = [tid, date_last, 0]
        df_opt_result_tid['lag_date'] = df_opt_result_tid.date.shift(1)
        df_opt_result_tid['lag_date'] = df_opt_result_tid['lag_date'].fillna(date_first)
        df_opt_result_tid['diff_date'] = (df_opt_result_tid['date'] - df_opt_result_tid['lag_date']).apply(lambda x: x.days)
        assert df_opt_result_tid.diff_date.max() <= params_dict['max_days_inc'] + 1 , f'alarm, for tid={tid} max_days for auto = {df_opt_result_tid.diff_date.max()}'



def check_over_balance(df_opt_result,  data, list_TID_cluster = None):
    """
    Считаем баланc банкоматов на утро после инкасации и проверяем не переполнены ли они
    """
    df_money_in = data.get_money_in()
    df_money_start = data.get_money_start()

    if list_TID_cluster is not None:
            df_money_in = df_money_in[df_money_in.TID.isin(list_TID_cluster)]
            df_money_start = df_money_start[df_money_start.TID.isin(list_TID_cluster)]

    params_dict = data.get_params_dict()
    df_money = pd.DataFrame(index=df_money_in['TID'].unique())
    df_money_start = df_money_start.set_index('TID').copy()
    df_money[df_money_in.date.min()] = 0
    for date in df_money_in.date.unique()[:-1]:
        df_money[date+timedelta(days=1)] = df_money_in[df_money_in.date==date].set_index('TID')['money_in']

    df_money.columns = pd.to_datetime(df_money.columns)

    df_money_sum = df_money.copy()

    for i in tqdm(df_money.columns):
        df_money_sum[i] = df_money[i] + df_money_start['money']
    for (j, column) in enumerate(df_money.columns):
        for column_2 in df_money.columns[j+1:]:
            df_money_sum[column_2] = df_money_sum[column_2] + df_money[column]

    df_money_sum = df_money_sum[df_money_sum.columns.sort_values()]


    unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
    dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
    dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}

    # обработаем инкасации
    list_date = np.sort(df_opt_result.date.unique())
    for day in tqdm(list_date):
        day = pd.to_datetime(day)
        for j in range(df_money_sum.shape[1]-1, dict_date_to_num[day]-1, -1):
            df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), dict_num_to_date[j]] -= df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), day]
    assert sum((df_money_sum > params_dict['max_money']).sum()) == 0, 'alarm, max_money in ATM is too big'

    return df_money_sum       