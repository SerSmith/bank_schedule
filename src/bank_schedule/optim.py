
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from pyomo.core.util import quicksum

import pandas as pd
from datetime import date, datetime, timedelta
import itertools
import numpy as np

from bank_schedule import dataclaster

from tqdm import tqdm


class OptModel:
    def __init__(self, data) -> None:
        self.model = pyo.ConcreteModel()
        self.data = data


    def add_basic_conceptions(self, money_start, days_from_inc, date_from, date_to, cluster):

        money_in = self.data.get_money_in(cluster)

    # date(2022, 11, 14)
        money_in = money_in[(money_in["date"].dt.date >= date_from) & (money_in["date"].dt.date <= date_to)]

        date_num_dict = {date: num for num, date in enumerate(sorted(money_in['date'].unique()))}
        money_in['date'] = money_in['date'].map(date_num_dict)

        params = self.data.get_params_dict()

        self.model.TIDS = pyo.Set(initialize=money_in['TID'].unique())

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

        self.model.TIDS_with_null = pyo.Set(initialize=np.append(money_in['TID'].unique(), -1))

        self.model.MINUTES_CARS = (datetime.combine(date.today(), params['day_end'] ) - \
                            datetime.combine(date.today(), params['day_start']))\
                            .total_seconds() / 60

        distance_matrix = self.data.get_distance_matrix(cluster)
        self.model.distance_matrix_dict = distance_matrix.set_index(["Origin_tid",	"Destination_tid"]).to_dict()['Total_Time']

        self.model.days_from_inc_dict = days_from_inc.set_index(["TID"]).to_dict()['days_from_inc']
    
        self.model.money_in_dict = money_in.set_index(['TID', 'date']).to_dict()['money_in']

        self.model.money_start_dict = {row[1]['TID']: row[1]['money'] for row in money_start.iterrows()}

        self.model.MAX_DATE = max(self.model.DATES)

        weights = self.calc_weights(money_start, days_from_inc, params)

        self.model.weights_dict = weights.set['TID'].to_dict()["weight"]


        # Переменные
        # Прогноз инкассации
        self.model.money_inc = pyo.Var(self.model.TIDS, self.model.DATES, within=pyo.Binary, initialize=0)



        # Затраты на инкассацию
        # model.costs_for_inc = pyo.Var(model.TIDS, model.DATES,  within=pyo.NonNegativeReals, initialize=0 )
        # def con_costs_for_inc_1(model, tid, date):
        #     return model.money_inc[tid, date] * model.COST_INC_MIN <= model.costs_for_inc[tid, date]
        # model.con_costs_for_inc_1 = pyo.Constraint(model.TIDS, model.DATES, rule=con_costs_for_inc_1 )

        # def con_costs_for_inc_2(model, tid, date):
        #     return model.money_inside_TID[tid, date] * model.COST_INC_PERS -  model.money_inc[tid, date] * M <= model.costs_for_inc[tid, date]
        # model.con_costs_for_inc_2 = pyo.Constraint(model.TIDS, model.DATES, rule=con_costs_for_inc_2 )

        #затраты на инкассацию
        def costs_for_inc(model, tid, date):
            return model.money_inc[tid, date] * model.COST_INC_MIN
        self.model.costs_for_inc = pyo.Expression(self.model.TIDS, self.model.DATES, rule=costs_for_inc )


        #Последовательность маршрута
        self.model.route = pyo.Var(self.model.TIDS_with_null, self.model.TIDS_with_null, self.model.DATES, within=pyo.Binary, initialize=0 )

        def con_route_1(model, tid, date):
            return quicksum([model.route[tid, tid2, date] for tid2 in model.TIDS_with_null if tid != tid2]) == model.money_inc[tid, date]
        self.model.con_route_1 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_route_1)

        def con_route_2(model, tid, date):
            return quicksum([model.route[tid2, tid, date] for tid2 in model.TIDS_with_null if tid != tid2]) == model.money_inc[tid, date]
        self.model.con_route_2 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_route_2 )

        def con_route_3(model, date):
            return quicksum([model.route[tid2, -1, date] for tid2 in model.TIDS]) == 1
        self.model.con_route_3 = pyo.Constraint(self.model.DATES, rule=con_route_3 )


        def con_route_4(model, date):
            return quicksum([model.route[-1, tid2, date] for tid2 in model.TIDS]) == 1
        self.model.con_route_4 = pyo.Constraint(self.model.DATES, rule=con_route_4 )


        # def con_route_2(model, tid1, tid2, date):
        #     return 1 - (model.money_inc[tid1, date] + model.money_inc[tid2, date]) * M <= model.route[tid1, tid2, date]
        # model.con_route_2 = pyo.Constraint(TIDS, TIDS, DATES, rule=con_route_2 )




        def con_max_time(model, date):
            return quicksum([model.route[tid1, tid2, date] * model.distance_matrix_dict[(tid1, tid2)] for tid1, tid2 in itertools.product(list(model.TIDS), list(model.TIDS)) if tid1 != tid2]) +\
            quicksum([model.money_inc[tid, date] * model.MIN_WAIT for tid in model.TIDS])\
            <= model.MINUTES_CARS
        self.model.con_max_time = pyo.Constraint(self.model.DATES, rule=con_max_time )



        def con_max_inc(model, date):
            return quicksum([model.money_inc[tid, date] for tid in model.TIDS]) <= 30
        self.model.con_max_inc = pyo.Constraint(self.model.DATES, rule=con_max_inc)

    



#заставляем иметь один цикл
        self.model.rank = pyo.Var(self.model.TIDS, self.model.DATES, within=pyo.NonNegativeReals, initialize=0)
        def rank1(model, tid1, tid2, date):
            return model.rank[tid1, date] + 1 <= model.rank[tid2, date] + model.M * (1 - model.route[tid1, tid2, date])
            # return model.rank[tid1, date] + model.route[tid1, tid2, date] <= model.rank[tid2, date]

        self.model.rank1 = pyo.Constraint(self.model.TIDS, self.model.TIDS, self.model.DATES, rule=rank1)



        return self.model


    def fixed_some_TID(self, list_TID_to_inc, list_TID_not_inc):

        for TID in list_TID_to_inc:
            self.model.money_inc[TID, 0].fix(1)


        for TID in list_TID_not_inc:
            self.model.money_inc[TID, 0].fix(0)


    def add_gready_concepts(self):

        self.model.OBJ = pyo.Objective(expr=
                                    quicksum([(self.model.weights_dict[tid]) * self.model.money_inc[(tid, date)] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES))])
                                    , sense=pyo.maximize)
        




    def add_full_optim_concepts(self):

        self.model.money_inside_TID = pyo.Var(self.model.TIDS, self.model.DATES,  within=pyo.NonNegativeReals, initialize=0 )

        # def con_days_from_inc(model, tid, date):
        #     right_date_border = date + model.MAX_DAYS_INC - max(model.days_from_inc_dict[tid] - date, 0)

        #     if right_date_border <= model.MAX_DATE + 1:
        #         out = (quicksum([ model.money_inc[tid, date_iter] for date_iter in  range(date, right_date_border)]) >= 1)
        #     else:
        #         out = pyo.Constraint.Feasible
        #     return out

        # self.model.con_days_from_inc = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_days_from_inc)

        def costs_from_money(model, tid, date):
            return model.money_inside_TID[tid, date] * model.OVERNIGHT_BY_DAY
        self.model.costs_from_money = pyo.Expression(self.model.TIDS, self.model.DATES, rule=costs_from_money )


        def con_money_inside_TID_1(model, tid, date):
            if date == 0:
                # TODO Проверить не теряем ли мы день
                out = model.money_start_dict[tid]
            else:
                out = model.money_inside_TID[tid, date - 1] + model.money_in_dict[(tid, date - 1)]
            return  out -  model.money_inc[tid, date] * model.M <= model.money_inside_TID[tid, date]
        self.model.con_money_inside_TID_1 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_money_inside_TID_1)



        def con_money_inside_TID_2(model, tid, date):
            return model.money_inside_TID[tid, date] <= (1 - model.money_inc[tid, date]) * self.model.M
        self.model.con_money_inside_TID_2 = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_money_inside_TID_2)


        # def con_max_money(model, tid, date):
        #     return model.money_inside_TID[tid, date] <= model.MAX_MONEY
        # self.model.con_max_money = pyo.Constraint(self.model.TIDS, self.model.DATES, rule=con_max_money)







        self.model.OBJ = pyo.Objective(expr=
                                    # quicksum([model.money_inside_TID[tid, date] for tid, date in itertools.product(list(model.TIDS), list(model.DATES)) ]) +
                                    # quicksum([model.days_from_inc[tid, date] for tid, date in itertools.product(list(model.TIDS), list(model.DATES)) ]) +
                                    (1/self.data.get_params_dict()['technical_cost']) * quicksum([self.model.costs_from_money[tid, date] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES)) ]) +
                                    (1/self.data.get_params_dict()['technical_cost']) * quicksum([self.model.costs_for_inc[tid, date] for tid, date in itertools.product(list(self.model.TIDS), list(self.model.DATES)) ]), sense=pyo.minimize)

    def load_presolve(self, presolved_models):

        for num, presolved_model in enumerate(presolved_models):       
            for tid in itertools.product(list(presolved_model.TIDS)):
                self.model.money_inc[tid, num].value = presolved_model.money_inc[tid, 0].value

            for tid1, tid2 in itertools.product(list(presolved_model.TIDS_with_null), list(presolved_model.TIDS_with_null)):
                self.model.route[tid1, tid2, num].value = presolved_model.route[tid1, tid2, 0].value
            


    def solve(self, optim_options):

        opt = SolverFactory('cbc')

        for key in optim_options:
            opt.options[key] = optim_options[key]

        results = opt.solve(self.model)

        print(results['Problem'])
        print(results['Solver'])
        return self.model, results
    
    @staticmethod
    def calc_weights(money_start, days_from_inc, params):
        weights = money_start.merge(days_from_inc, on='TID')
        weights['weight'] = weights["money"] + weights["days_from_inc"] ** params["pow_weight"]
        return weights[["TID", 'weight']]



def update_money_start(money_start, money_in_full, now_date, incs):

    money_in = money_in_full.loc[money_in_full["date"].apply(lambda x: x.strftime('%Y-%m-%d')) == str(now_date), ["TID",	"money_in"]]

    all_actions = money_start.merge(money_in, on=["TID"]).merge(incs, on=["TID"])

    all_actions["new_money_start"] = all_actions["money"] * (1 - all_actions["inc"]) + all_actions["money_in"]

    out = all_actions[["TID", "new_money_start"]].rename(columns={"new_money_start": "money"})

    return out

def update_days_from_inc(days_from_inc, incs):

    out = days_from_inc.merge(incs, on=["TID"])

    
    out['days_from_inc'] = out['days_from_inc'] * (1 - out["inc"]) + 1

    return out[["TID", 'days_from_inc']]

def calc_inc(opt):

    solution_dict = opt.money_inc.extract_values()


    solution_pd = pd.DataFrame(solution_dict.items(), columns=['index', 'inc'])
    solution_pd['TID'], solution_pd['date_num'] = zip(*solution_pd['index'])

    solution_pd = solution_pd[['TID', 'inc']]
    return solution_pd


def find_TID_for_inc(money_start, data,  days_from_inc, days_for_inc_force = [13, 14]):
    params_dict = data.get_params_dict()
    list_TID_max_money = list(money_start[money_start['money'] > params_dict['max_money'] ].TID.unique()) 

    list_TID_from_inc = list(days_from_inc[days_from_inc.days_from_inc.isin(days_for_inc_force)].TID.unique())
    list_TID = list(set(list_TID_max_money) | set(list_TID_from_inc))
    return list_TID

def find_TID_not_inc(days_from_inc, days_for_not_inc = []):
    list_TID_not_inc = list(days_from_inc[days_from_inc.days_from_inc.isin(days_for_not_inc)].TID.unique())
    return list_TID_not_inc


def presolve(data, date_from, day_count, cluster_num):

    models = []
    results = []

    money_start = data.get_money_start()
    money_in = data.get_money_in(cluster_num)

    
    tids_by_cluster = data.get_tids_by_claster(cluster_num)

    days_from_inc = pd.DataFrame(tids_by_cluster, columns=['TID'])

    days_from_inc['days_from_inc'] = 0



    for now_date in tqdm([date_from + timedelta(days=n) for n in range(day_count)], desc=f"presolve cluster_num={cluster_num}"):

        optim = OptModel(data)

        optim.add_basic_conceptions(money_start, days_from_inc, now_date, now_date, cluster_num)
        optim.add_gready_concepts()        
        list_TID_to_inc = find_TID_for_inc(money_start, data, days_from_inc) #, days_for_inc_force)
        list_TID_not_inc = find_TID_not_inc(days_from_inc) #, days_for_not_inc)
        optim.fixed_some_TID(list_TID_to_inc, list_TID_not_inc) 

        optim_options = {
            'ratioGap': 0.01, 
            'sec': 600,
            # 'threads': -1
            }

        model, result = optim.solve(optim_options)
        models.append(model)
        results.append(result)

        incs = calc_inc(model)
        money_start = update_money_start(money_start, money_in, now_date, incs)
        days_from_inc = update_days_from_inc(days_from_inc, incs)

    return models, results


if __name__ == "__main__":

    cluster_num = 3

    data = dataclaster.DataClaster('/Users/sykuznetsov/Documents/GitHub/bank_schedule/data/raw')
    data.run_cluster(0.1, 5)


    # presolved_models, presolved_results = presolve(data, date(2022, 11, 1), 15, cluster_num)








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