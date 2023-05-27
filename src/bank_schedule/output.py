from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# def find_ATM_balance_evening(df_opt_result, df_money, df_money_start, params_dict):
#     """
#     Функция принимает результаты оптимизации и выдает датафрейм с остатками на счетах в банкоматах на каждый день
#     """
#     df_money = df_money.copy()
#     df_money_start = df_money_start.set_index('TID').copy()
#     df_money = df_money.set_index('TID')
#     df_money.drop(['остаток на 31.08.2022 (входящий)'], axis=1, inplace=True)
#     df_money.columns = pd.to_datetime(df_money.columns)

#     df_money_sum = df_money.copy()

#     for i in tqdm(df_money.columns):
#         df_money_sum[i] = df_money[i] + df_money_start['money']
#     for (j, column) in enumerate(df_money.columns):
#         for column_2 in df_money.columns[j+1:]:
#             df_money_sum[column_2] = df_money_sum[column_2] + df_money[column]

#     date_start = df_money_sum.columns.min()
#     date_minus_1  = date_start - timedelta(days=1)
#     df_money_sum[date_minus_1] = df_money_start['money']
#     df_money_sum = df_money_sum[df_money_sum.columns.sort_values()]


#     unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
#     dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
#     dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}

#     # обработаем инкасации
#     for day in tqdm(unique_date):
#         date_before = dict_num_to_date[dict_date_to_num[day]-1]
#         for j in range(dict_date_to_num[day], df_money_sum.shape[1]):
#             df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), dict_num_to_date[j]] -= df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), date_before]

#     df_money_sum.drop([date_minus_1], axis=1, inplace=True)

#     # ограничиваем максимальной вместительностью
#     condition = df_money_sum < params_dict['max_money']
#     df_money_sum = df_money_sum.where(condition, other = params_dict['max_money'])
#     return df_money_sum

def find_ATM_balance_evening(df_opt_result, data, list_TID_claster = None):
    """
    Функция принимает результаты оптимизации и выдает датафрейм с остатками на счетах в банкоматах на каждый день
    """
    params_dict = data.get_params_dict()
    df_money_in = data.get_money_in()
    df_money_start = data.get_money_start()

    if list_TID_claster is not None:
        df_money_in = df_money_in[df_money_in.TID.isin(list_TID_claster)]
        df_money_start = df_money_start[df_money_start.TID.isin(list_TID_claster)]

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


    #unique_date = pd.to_datetime(df_opt_result.date.unique())     #уникальные дни инкасации
    dict_date_to_num = { k:v for (v, k) in enumerate(df_money_sum.columns)}
    dict_num_to_date= { v:k for (v, k) in enumerate(df_money_sum.columns)}

    # обработаем инкасации
    list_date = np.sort(df_opt_result.date.unique())
    for day in tqdm(list_date):
        day = pd.to_datetime(day)
        date_before = dict_num_to_date[dict_date_to_num[day]-1]
        for j in range(dict_date_to_num[day], df_money_sum.shape[1]):
            df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), dict_num_to_date[j]] -= df_money_sum.loc[list(df_opt_result[df_opt_result.date==day]['TID']), date_before]

    df_money_sum.drop([date_minus_1], axis=1, inplace=True)

    # ограничиваем максимальной вместительностью
    #condition = df_money_sum < params_dict['max_money']
    #df_money_sum = df_money_sum.where(condition, other = params_dict['max_money'])
    return df_money_sum


def find_cost_inc(df_money_sum, df_opt_result, data, list_TID_claster = None):
    """
    Функция принимающая на вход остатки в банкоматах и результат оптимизации 
    и выдающая стоимость инкасаций, которые мы нашли в результате оптимизаций
    """
    df_money_start = data.get_money_start()
    params_dict = data.get_params_dict()

    if list_TID_claster is not None:
        df_money_start = df_money_start[df_money_start.TID.isin(list_TID_claster)]


    date_start = df_money_sum.columns.min()
    date_minus_1  = date_start - timedelta(days=1)
    df_money_sum[date_minus_1] = list(df_money_start['money'])
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

def find_sum_fond(df_opt_result,  data, list_TID_claster = None):
    df_money_in = data.get_money_in()
    df_money_start = data.get_money_start()
    if list_TID_claster is not None:
        df_money_in = df_money_in[df_money_in.TID.isin(list_TID_claster)]
        df_money_start = df_money_start[df_money_start.TID.isin(list_TID_claster)]

    params_dict = data.get_params_dict()
    "Считаем стоимость фондирования (смотрим от суммы на утро после инкасирования)"
    df_money_sum = find_ATM_balance_morning(df_opt_result, data, list_TID_claster)
    df_fond = df_money_sum * params_dict['overnight'] / 100 / 365
    return df_fond

def find_all_cost(df_money_sum, df_opt_result,data, list_TID_claster = None ):
    """
    """
    params_dict = data.get_params_dict()
    df_money_start = data.get_money_start()
    if list_TID_claster is not None:
        df_money_start = df_money_start[df_money_start.TID.isin(list_TID_claster)]
    df_cost_inc = find_cost_inc(df_money_sum, df_opt_result, data, list_TID_claster)
    df_fond = find_sum_fond(df_opt_result, data, list_TID_claster)
    cost_auto = pd.DataFrame([params_dict['price_car'] * len(df_opt_result.auto.unique())] * ((df_fond.columns.max() - df_fond.columns.min() ).days + 1 ), index = df_fond.columns) # или по датам лучше переписать на df_money_in ?    all_cost = sum(df_fond.add(df_cost_inc, fill_value=0).sum()) + cost_auto

    df_cost_inc_by_days = df_cost_inc.sum()
    df_fond_by_days = df_fond.sum()
    all_cost_by_days = pd.concat([df_cost_inc_by_days, df_fond_by_days, cost_auto], axis=1)
    all_cost_by_days.columns = ['Затраты на инкасацию','Затраты на фондирование','Затраты на машины']
    return all_cost_by_days

def find_ATM_balance_morning(df_opt_result,  data, list_TID_claster=None):
    "Считаем баланыс банкоматов на утро после инкасации"
    df_money_in = data.get_money_in()
    df_money_start = data.get_money_start()

    if list_TID_claster is not None:
        df_money_in = df_money_in[df_money_in.TID.isin(list_TID_claster)]
        df_money_start = df_money_start[df_money_start.TID.isin(list_TID_claster)]

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
    
    return df_money_sum

def find_routes_check(obj, data):
    df = pd.DataFrame(columns=['rebro', 'rebro_flg','date_int'])
    list_df = []
    for auto in range(len(obj)):
        for day in range(len(obj[auto][0])):    
            route_dict = obj[0][0][day][0].route.extract_values()
            df_part = pd.DataFrame(route_dict.items(), columns=['rebro', 'rebro_flg'])
            #df_part['date_int'] = day
            df_part['auto'] = auto 
            df_part['date'] = obj[0][0][day][1]
            df = pd.concat([df, df_part])
            
    df['date'] = pd.to_datetime(df['date'])
    df['ATM1'] = df['rebro'].apply(lambda x: x[0])
    df['ATM2'] = df['rebro'].apply(lambda x: x[1])

    #df['date'] = df['date_int'].apply(lambda x: dict_num_to_date[x])
    #dict_date_to_num = { k:v for (v, k) in enumerate(np.sort(list(df_money_sum.columns)))}
    #dict_num_to_date= { v:k for (v, k) in enumerate(np.sort(list(df_money_sum.columns)))}

    params_dict = data.get_params_dict()
    start_time = params_dict['day_start']
    end_time = params_dict['day_end']
    min_wait = params_dict['min_wait']
    df_routes = pd.DataFrame(columns = ['auto','TID','start_time', 'end_time', 'date'])

    for date in list(df[(df.rebro_flg>0)].date.unique()):
        #print('date', date)
        #df_rebro_date = df[(df.date_int==date) & (df.rebro_flg>0)]
        auto_list = list(df[(df.date==date) & (df.rebro_flg>0)].auto.unique())
        for auto in auto_list:
            #print(auto, date)
            df_rebro_date_auto = df[(df.date==date) & (df.auto==auto) & (df.rebro_flg>0)]
            
            current_ATM = df_rebro_date_auto[df_rebro_date_auto.ATM1==-1]['ATM2'].values[0]
            list_ATM = []
            sum_time = 0
            start_datetime = date + timedelta(hours = start_time.hour, minutes = start_time.minute)
            end_datetime = date + timedelta(hours = end_time.hour, minutes = end_time.minute)
            current_datetime = start_datetime
            while current_ATM!=-1:
                #print(current_ATM)
                list_ATM.append(current_ATM)
                next_ATM = df_rebro_date_auto[df_rebro_date_auto.ATM1==current_ATM]['ATM2'].values[0]

                end_ATM_datetime = current_datetime + timedelta(minutes = min_wait)
                if next_ATM!=-1:
                    next_ATM_start_datetime = end_ATM_datetime + timedelta(minutes=df_distance_matrix[(df_distance_matrix.Origin_tid==current_ATM) & (df_distance_matrix.Destination_tid==next_ATM)]['Total_Time'].values[0])
                df_routes.loc[df_routes.shape[0]] = [auto, current_ATM, current_datetime, end_ATM_datetime, date]

                assert  end_ATM_datetime<=end_datetime, f'alarm, for date={dict_num_to_date[date]} and auto={auto} end_time = {end_ATM_datetime} is bigger then {end_datetime}'
                current_ATM = next_ATM
                current_datetime = next_ATM_start_datetime    
    df_opt_result = df_routes[['TID','date','auto']]
    return df_opt_result