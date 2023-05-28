
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

from pyomo.core.util import quicksum

import pandas as pd
from datetime import date, datetime, timedelta
import itertools
import numpy as np

from bank_schedule import dataclaster

from tqdm import tqdm
import math


class OptModel:
    """Класс, реализующий оптимизационную задачу через MILP
    """
    def __init__(self, data) -> None:
        """_summary_

        Args:
            data (DataClaster): _description_
        """
        self.model = pyo.ConcreteModel()
        self.data = data


    def  add_basic_conceptions(self,
                               tids,
                               money_start,
                               days_from_inc,
                               date_from,
                               date_to,
                               cluster):
        """Добавление в оптимизацию сущностей и ограничений,которые можно использовать
           как при подневной оптимизации, так и при глобальной

        Args:
            tids (list[int]): TID id, для которых бедт производиться отпимизация
            money_start (pd.DataFrame): Остатки  банкоматах на день до оптимизационногопериода
            days_from_inc (pd.DataFrame): Количечество дней, которое банкомат не пополнялся
            date_from (datetime): Дата начала оптимизируемого периода
            date_to (datetime): Дата окончания оптимизируемого периода
            cluster (int): Номер кластера
        """


        money_in = self.data.get_money_in(cluster)

        money_in = money_in[(money_in["date"].dt.date >= date_from) & (money_in["date"].dt.date <= date_to)]

        date_num_dict = {date: num for num, date in enumerate(sorted(money_in['date'].unique()))}
        money_in['date'] = money_in['date'].map(date_num_dict)

        params = self.data.get_params_dict()

        self.model.TIDS = pyo.Set(initialize=tids)

        self.model.DATES = pyo.Set(initialize=money_in['date'].unique())

        self.model.MAX_MONEY = params['max_money']

        self.model.MAX_DAYS_INC = params['max_days_inc']

        self.model.OVERNIGHT_BY_DAY = params['overnight']/100/365

        self.model.COST_INC_PERS = params['cost_inc_pers']

        self.model.COST_INC_MIN = params['cost_inc_min']

        self.model.MIN_WAIT = params["min_wait"]

        self.model.POW_WEIGHT = params["pow_weight"]

        self.model.MAX_TIDS_BY_DAY = params["max_tids_by_day"]

        self.model.M = params["M"]

        self.model.TIDS_with_null = pyo.Set(initialize=np.append(tids, -1))

        self.model.MINUTES_CARS = (datetime.combine(date.today(), params['day_end']) - \
                                   datetime.combine(date.today(), params['day_start']))\
                                   .total_seconds() / 60

        distance_matrix = self.data.get_distance_matrix(cluster)

        self.model.distance_matrix_dict = distance_matrix.set_index(["Origin_tid",	"Destination_tid"]).to_dict()['Total_Time']

        self.model.days_from_inc_dict = days_from_inc.set_index(["TID"]).to_dict()['days_from_inc']
    
        self.model.money_in_dict = money_in.set_index(['TID', 'date']).to_dict()['money_in']

        self.model.money_start_dict = {row[1]['TID']: row[1]['money'] for row in money_start.iterrows()}

        self.model.MAX_DATE = max(self.model.DATES)

        self.model.MAX_TIDS_BY_DAY = calc_max_tid_by_date(len(self.model.TIDS), params)

        weights = self.calc_weights(money_start, days_from_inc, params)

        self.model.weights_dict = weights.set_index('TID').to_dict()["weight"]


        # Переменные
        # Налчие инкассации банкомата tid на дату date
        self.model.money_inc = pyo.Var(self.model.TIDS, self.model.DATES, within=pyo.Binary, initialize=0)
 

        #Последовательность маршрута
        #Имеется ли ребро между банкоматами в дату date. Также мы добавляем "виртуальный" банкомат -1, расстояние до которого всегда 0.
        # Он нужен для того, что бы не обрабатывать правила граничный точчек - благодаря банкомату -1 у нас для каждой вершины есть путь в
        # вершину и путь и из вершины.
        self.model.route = pyo.Var(self.model.TIDS_with_null, self.model.TIDS_with_null, self.model.DATES, within=pyo.Binary, initialize=0 )

        # Если и только если мы приехали в банкомат для него должно быть ребро - начало
        def con_route_1(model, tid, date):
            return quicksum([model.route[tid, tid2, date] for tid2 in model.TIDS_with_null if tid != tid2]) == model.money_inc[tid, date]
        self.model.con_route_1 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_route_1)

        # Если и только если мы приехали в банкомат для него должно быть ребро - конца
        def con_route_2(model, tid, date):
            return quicksum([model.route[tid2, tid, date] for tid2 in model.TIDS_with_null if tid != tid2]) == model.money_inc[tid, date]
        self.model.con_route_2 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_route_2)

        # Мы обязаны откуда-то стартовать
        def con_route_3(model, date):
            return quicksum([model.route[tid2, -1, date] for tid2 in model.TIDS]) == 1
        self.model.con_route_3 = pyo.Constraint(self.model.DATES, rule=con_route_3 )

        # Мы обязаны когда-то закончить
        def con_route_4(model, date):
            return quicksum([model.route[-1, tid2, date] for tid2 in model.TIDS]) == 1
        self.model.con_route_4 = pyo.Constraint(self.model.DATES, rule=con_route_4 )

        # Мы должны успеть посетить все банкоматы маршрута за время дня
        def con_max_time(model, date):
            return quicksum([model.route[tid1, tid2, date] * model.distance_matrix_dict[(tid1, tid2)] for tid1, tid2 in itertools.product(list(model.TIDS), list(model.TIDS)) if tid1 != tid2]) +\
            quicksum([model.money_inc[tid, date] * model.MIN_WAIT for tid in model.TIDS])\
            <= model.MINUTES_CARS
        self.model.con_max_time = pyo.Constraint(self.model.DATES, rule=con_max_time )

        # Облегчим задачу оптимизации - эврестически ограничив максимальную длину маршрута
        def con_max_inc(model, date):
            return quicksum([model.money_inc[tid, date] for tid in model.TIDS]) <= model.MAX_TIDS_BY_DAY
        self.model.con_max_inc = pyo.Constraint(self.model.DATES, rule=con_max_inc)

    

        # Запрещаем все циклы, кроме проходящего через -1
        self.model.rank = pyo.Var(self.model.TIDS, self.model.DATES, within=pyo.NonNegativeReals, initialize=0)
        def rank1(model, tid1, tid2, date):
            return model.rank[tid1, date] + 1 <= model.rank[tid2, date] + model.M * (1 - model.route[tid1, tid2, date])
        self.model.rank1 = pyo.Constraint(self.model.TIDS, self.model.TIDS, self.model.DATES, rule=rank1)

        return self.model


    def fixed_some_TID(self, list_TID_to_inc, list_TID_not_inc):
        """Функция, фиксирующая банкоматы, которые мы обязаны посетить и те которые мы обязаны не посещеать
        Args:
            list_TID_to_inc (list(int)): Банкоматы, которые обязаны посетить
            list_TID_not_inc (list(int)): Банкоматы, которые обязаны не посещать
        """

        for TID in list_TID_to_inc:
            self.model.money_inc[TID, 0].fix(1)


        for TID in list_TID_not_inc:
            self.model.money_inc[TID, 0].fix(0)
        

    def add_gready_concepts(self):
        """Добаляем целевую для подневной оптимизации
        """
        self.model.OBJ = pyo.Objective(expr=
                                    quicksum([(self.model.weights_dict[tid]) * self.model.money_inc[(tid, date)] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES))])
                                    , sense=pyo.maximize)
        




    def add_full_optim_concepts(self):
        """Добавление ксловий для полной оптимизации
        """

        #затраты на инкассацию
        def costs_for_inc(model, tid, date):
            return model.money_inc[tid, date] * model.COST_INC_MIN
        self.model.costs_for_inc = pyo.Expression(self.model.TIDS, self.model.DATES, rule=costs_for_inc )

        self.model.money_inside_TID = pyo.Var(self.model.TIDS, self.model.DATES,  within=pyo.NonNegativeReals, initialize=0 )

        # def con_days_from_inc(model, tid, date):
        #     right_date_border = date + model.MAX_DAYS_INC - max(model.days_from_inc_dict[tid] - date, 0)

        #     if right_date_border <= model.MAX_DATE + 1:
        #         out = (quicksum([ model.money_inc[tid, date_iter] for date_iter in  range(date, right_date_border)]) >= 1)
        #     else:
        #         out = pyo.Constraint.Feasible
        #     return out

        # self.model.con_days_from_inc = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_days_from_inc)

        # Затраты на остатки в банкоматах 
        def costs_from_money(model, tid, date):
            return model.money_inside_TID[tid, date] * model.OVERNIGHT_BY_DAY
        self.model.costs_from_money = pyo.Expression(self.model.TIDS, self.model.DATES, rule=costs_from_money )


        def con_money_inside_TID_1(model, tid, date):
            if date == 0:
                out = model.money_start_dict[tid]
            else:
                out = model.money_inside_TID[tid, date - 1] + model.money_in_dict[(tid, date - 1)]
            return  out -  model.money_inc[tid, date] * model.M <= model.money_inside_TID[tid, date]
        self.model.con_money_inside_TID_1 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_money_inside_TID_1)



        def con_money_inside_TID_2(model, tid, date):
            return model.money_inside_TID[tid, date] <= (1 - model.money_inc[tid, date]) * self.model.M
        self.model.con_money_inside_TID_2 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_money_inside_TID_2)

        # Запрещено иметь деньги в банкомате вышк определенной суммы
        def con_max_money(model, tid, date):
            return model.money_inside_TID[tid, date] <= model.MAX_MONEY
        self.model.con_max_money = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_max_money)




        self.model.OBJ = pyo.Objective(expr= 
                                    quicksum([self.model.costs_from_money[tid, date] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES)) ]) +
                                    quicksum([self.model.costs_for_inc[tid, date] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES)) ])
                                    ,
                                    sense=pyo.minimize)

    def load_presolve(self, presolved_models):
        """Функция 

        Args:
            presolved_models (_type_): _description_
        """
        for num, presolved_model in enumerate(presolved_models):       
            for tid in itertools.product(list(presolved_model.TIDS)):
                self.model.money_inc[tid, num].value = presolved_model.money_inc[tid, 0].value

            for tid1, tid2 in itertools.product(list(presolved_model.TIDS_with_null), list(presolved_model.TIDS_with_null)):
                self.model.route[tid1, tid2, num].value = presolved_model.route[tid1, tid2, 0].value
            


    def solve(self, optim_options):

        opt = SolverFactory('cbc')

        for key in optim_options:
            opt.options[key] = optim_options[key]

        results = opt.solve(self.model, tee=True)

        print(results['Problem'])
        print(results['Solver'])
        return self.model, results
    
    @staticmethod
    def calc_weights(money_start, days_from_inc, params):
        weights = money_start.merge(days_from_inc, on='TID')
        # weights['weight'] = weights["money"] + weights["days_from_inc"] ** params["pow_weight"]
        weights['weight'] = weights["money"] + 100000 * weights["days_from_inc"] ** 2
        return weights[["TID", 'weight']]


    def get_top_tids(self, quant, money_start, days_from_inc, data):
        params = data.get_params_dict()
        weights = self.calc_weights(money_start, days_from_inc, params)
        return weights.sort_values('weight', ascending=False)['TID'].head(quant).to_list()


def calc_max_tid_by_date(tids_quant, params, add_coeff=1.2):
    return math.ceil(tids_quant / params['max_days_inc'] * add_coeff)

def update_money_start(money_start, money_in_full, now_date, incs):

    money_in = money_in_full.loc[money_in_full["date"].apply(lambda x: x.strftime('%Y-%m-%d')) == str(now_date), ["TID",	"money_in"]]

    all_actions = money_start.merge(money_in, on=["TID"], how='left').merge(incs, on=["TID"], how='left')

    all_actions = all_actions.fillna(0)

    all_actions["new_money_start"] = all_actions["money"] * (1 - all_actions["inc"]) + all_actions["money_in"]

    out = all_actions[["TID", "new_money_start"]].rename(columns={"new_money_start": "money"})

    return out

def update_days_from_inc(days_from_inc, incs):

    out = days_from_inc.merge(incs, on=["TID"], how='left')

    out = out.fillna(0)
    
    out['days_from_inc'] = (out['days_from_inc'] + 1) * (1 - out["inc"])

    return out[["TID", 'days_from_inc']]

def calc_inc(opt):

    solution_dict = opt.money_inc.extract_values()


    solution_pd = pd.DataFrame(solution_dict.items(), columns=['index', 'inc'])
    solution_pd['TID'], solution_pd['date_num'] = zip(*solution_pd['index'])

    solution_pd = solution_pd[['TID', 'inc']]
    return solution_pd


def find_TID_for_inc(money_start, days_from_inc, data, days_for_inc_force=None):
    if days_for_inc_force is None:
        days_for_inc_force = [14]

    params_dict = data.get_params_dict()
    list_TID_max_money = list(money_start[money_start['money'] > params_dict['max_money'] ].TID.unique()) 

    list_TID_from_inc = list(days_from_inc[days_from_inc.days_from_inc.isin(days_for_inc_force)].TID.unique())
    list_TID = list(set(list_TID_max_money) | set(list_TID_from_inc))
    return list_TID

def find_TID_not_inc(days_from_inc, days_for_not_inc = []):
    list_TID_not_inc = list(days_from_inc[days_from_inc.days_from_inc.isin(days_for_not_inc)].TID.unique())
    return list_TID_not_inc




def presolve(data, date_from, day_count, top_tids_quant, cluster_num, tries, step):

    models = []
    results = []

    money_start = data.get_money_start(cluster_num)
    money_in = data.get_money_in(cluster_num)

    
    tids_by_cluster = data.get_tids_by_claster(cluster_num)

    days_from_inc = pd.DataFrame(tids_by_cluster, columns=['TID'])

    days_from_inc['days_from_inc'] = 0



    for now_date in tqdm([date_from + timedelta(days=n) for n in range(day_count)], desc=f"presolve cluster_num={cluster_num}"):

        for i in range(tries):
            optim = OptModel(data)

            tids = set(optim.get_top_tids(top_tids_quant + i * step, money_start, days_from_inc, data))

            tids_have_to_visit = find_TID_for_inc(money_start, days_from_inc, data)


            tids = tids.union(set(tids_have_to_visit))
            
            #print(f'all tids: {tids}')

            print(f"Обяззательных постоматов {len(tids_have_to_visit)}: {tids_have_to_visit}")
            print(f"Всего оптимизируемых банкоматов {len(tids)}")


            optim.add_basic_conceptions(list(tids), money_start, days_from_inc, now_date, now_date, cluster_num)
            optim.add_gready_concepts()
            optim.fixed_some_TID(tids_have_to_visit, [])

            optim_options = {
                'ratioGap': 0.15, 
                'sec': 300,
                'threads': 8,
                "heuristics": "on",
                "autoScale": 'on',
                "feaspump": "on",
                "greedyHeuristic": "on"
                }

            model, result = optim.solve(optim_options)
            if result['Solver'][0]['Termination condition']==TerminationCondition.optimal:
                print(f'find solution for {top_tids_quant + i * step}')
                break
        #assert result['Solver'][0]['Termination condition']==TerminationCondition.optimal   
        models.append([model, now_date])
        incs = calc_inc(model)

        results.append((result, incs, money_start, days_from_inc))


        money_start = update_money_start(money_start, money_in, now_date, incs)
        days_from_inc = update_days_from_inc(days_from_inc, incs)

    return models, results


if __name__ == "__main__":

    cluster_num = 3

    data = dataclaster.DataClaster('/Users/sykuznetsov/Documents/GitHub/bank_schedule/data/raw')
    data.run_cluster(0.1, 5)


    presolved_models, presolved_results = presolve(data, date(2022, 11, 1), 15, cluster_num)




# optim = OptModel(data)

# money_start = data.get_money_start()

# tids_by_cluster = data.get_tids_by_claster(cluster_num)

# days_from_inc = pd.DataFrame(tids_by_cluster, columns=['TID'])

# days_from_inc['days_from_inc'] = 0

# optim.add_basic_conceptions(money_start, days_from_inc, date(2022, 11, 1), date(2022, 11, 15), cluster_num)
# optim.add_full_optim_concepts()

# optim_options = {
#     'ratioGap': 0.0001, 
#     'sec': 120,
#     'threads': 10}

# optim.load_presolve(presolved_models)

# print("start solving")

# model = optim.solve(optim_options)