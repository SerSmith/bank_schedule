from tqdm import tqdm
from pandas as pd
from datetime import datetime, timedelta

def find_ATM_balance(df_opt_result, df_money, df_money_start, max_money):
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
    condition = df_money_sum < max_money
    df_money_sum = df_money_sum.where(condition, other = max_money)
    return df_money_sum