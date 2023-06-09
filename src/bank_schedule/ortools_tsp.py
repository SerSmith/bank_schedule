"""Simple Travelling Salesperson Problem (TSP) between cities."""

from typing import List, Tuple, Dict
from copy import deepcopy

import pandas as pd
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from bank_schedule.data import distances_matrix_from_dataframe
from bank_schedule.data import Data


def create_data_model(distance_matrix: List[List[int]],
                      depot: int=0):
    """Готовит данные для решения

    Args:
        distance_matrix (List[List[int]]): _description_
        depot (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    data = {}
    data['distance_matrix'] = distance_matrix.tolist()
    data['depot'] = depot
    return data


def get_route_and_time(manager, routing, solution) -> Tuple[List[int], List[float], float]:
    """Возвращает маршрут, время и суммарное время

    Args:
        manager (_type_): _description_
        routing (_type_): _description_
        solution (_type_): _description_

    Returns:
        Tuple[List[int], List[float], float]: _description_
    """

    index = routing.Start(0)

    route_time = []
    plan_output = []

    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_time.append(routing.GetArcCostForVehicle(previous_index, index, 0) / 100)

    plan_output.append(manager.IndexToNode(index))

    return plan_output, route_time, sum(route_time)


def print_solution(manager, routing, solution):
    """_summary_

    Args:
        manager (_type_): _description_
        routing (_type_): _description_
        solution (_type_): _description_
    """
    print(f'Objective: {solution.ObjectiveValue() / 100} minutes')
    index = routing.Start(0)
    plan_output = 'Route for vehicle:\n'
    route_time = 0
    while not routing.IsEnd(index):
        plan_output += f' {manager.IndexToNode(index)} ->'
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_time += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f' {manager.IndexToNode(index)}\n'
    plan_output += f'Route time: {route_time / 100} minutes\n'
    print(plan_output)


def solve_tsp_ortools(tids_list: List[int],
                      distances_df: pd.DataFrame,
                      depot: int,
                      verbose: bool=False) -> Tuple[List[int], List[float], float]:
    """_summary_

    Args:
        tids_list (List[int]): идентификаторы банкоматов
        distances_df (pd.DataFrame): датафрейм с расстояниями
        depot (int, optional): _description_. Defaults to 0.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[List[int], List[float], float]: _description_
    """
    depot = tids_list.index(depot)

    distance_matrix = distances_matrix_from_dataframe(distances_df,
                                                      tids_list=tids_list,
                                                      convert_to_int=True,
                                                      add_delay_time=True)
    # Instantiate the data problem.
    data = create_data_model(distance_matrix, depot)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           1,
                                           data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # PATH_CHEAPEST_ARC 4661.13 minutes
    # PATH_MOST_CONSTRAINED_ARC 4725.2 minutes
    # SAVINGS 4640.44 minutes
    # SWEEP No solution found !
    # CHRISTOFIDES 4584.5 minutes <---
    # BEST_INSERTION No solution found !
    # PARALLEL_CHEAPEST_INSERTION 4645.21 minutes
    # SEQUENTIAL_CHEAPEST_INSERTION 4645.21 minutes
    # LOCAL_CHEAPEST_INSERTION 4643.61 minutes
    # LOCAL_CHEAPEST_COST_INSERTION 4643.61 minutes
    # GLOBAL_CHEAPEST_ARC 4617.79 minutes
    # LOCAL_CHEAPEST_ARC 4578.29 minute <---
    # FIRST_UNBOUND_MIN_VALUE 4707.0 minutes
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # pylint: disable=no-member

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if verbose:
        if solution:
            print_solution(manager, routing, solution)
        else:
            print('No solution found !')

    if not solution:
        raise ValueError('No solution found !')

    steps, steps_times, summary_time = get_route_and_time(manager, routing, solution)
    steps = [tids_list[i] for i in steps]
    return steps[:-1], steps_times[:-1], summary_time


def get_best_route(loader: Data,
                   tids_list: List[int],
                   n_iterations: int=10) -> Tuple[List[int], List[int], float]:
    """Возвращает список банкоматов, которые нужно обслужить
    в порядке оптимальног времени в пути, времена убытия из них

    Args:
        tids_list (List[int]): список банкоматов для обслуживания
        distances_df (pd.DataFrame): датафрейм расстояний
        n_iterations (int, optional): сколько попыток построения маршрута сделать. Defaults to 10.

    Returns:
        List[int]: _description_
    """
    distances_df = loader.get_distance_matrix().copy()

    best_sum_time = float('inf')
    best_route = []
    best_route_times = []
    for _ in range(n_iterations):
        depot = np.random.choice(tids_list)
        route, route_times, sum_time = solve_tsp_ortools(tids_list,
                                               distances_df,
                                               depot)
        if len(route) != len(set(route)):
            print(route)
            print(pd.Series(data=route).value_counts())
            raise ValueError('Какие-то точки в маршруте посещаются не один раз')

        if sum_time < best_sum_time:
            best_sum_time = sum_time
            best_route = route
            best_route_times = route_times

    return best_route, best_route_times, best_sum_time


def optimize_routes(loader: Data,
                    schedule_df: pd.DataFrame,
                    n_iterations: int=1,
                    max_route_time: float=720.0,
                    verbose: bool=False) -> Tuple[Dict, Dict]:
    """Находит количество машин, необходимое для обслуживания банкоматов
    согласно расписанию

    Args:
        loader (Data): класс загрузки данных
        schedule_df (pd.DataFrame): датафрейм с расписанием инкассации. Должен содержать колонки
        'TID' и 'date'
        n_iterations (int, optional): количество итераций построения маршрута.
         Выбирается лучшее решение из полученных на каждой итерации Defaults to 1.
        max_route_time (float, optional): максимально допустимое время маршрута в минутах
         для одного броневика. Defaults to 720.0.

    Returns:
        Tuple[Dict, Dict: словари со списками банкоматов в порядке инкассации и временем
         пути + инкассации на каждую дату из расписания
    """

    schedule = schedule_df.groupby('date')['TID'].agg(list).to_dict()
    car_routes_dict = {}
    cars_routes_times_dict = {}

    for date, tids_list in schedule.items():

        car_routes_dict[date] = {}
        cars_routes_times_dict[date] = {}

        route, route_time, _ = get_best_route(loader, tids_list, n_iterations=n_iterations)

        if len(route) != len(set(route)):
            raise ValueError('В маршруте дублируются точки (есть петли)')

        cars_routes, cars_routes_times = split_route_by_cars(route, route_time, max_route_time)

        car_routes_dict[date] = cars_routes
        cars_routes_times_dict[date] = cars_routes_times

        max_rt, min_rt = -float('inf'), float('inf')

        for _car, _route_time in cars_routes_times.items():

            _route_time_sum = sum(_route_time)
            max_rt = max(max_rt, _route_time_sum)
            min_rt = min(min_rt, _route_time_sum)

            if _route_time_sum > max_route_time:
                raise ValueError(f'Время {sum(_route_time)} > {max_route_time} для машины {_car}')

        if verbose:
            print(f'date: {str(date)} | '
                  f'max route time: {round(max_rt, 2):<7}| '
                  f'max route time: {round(min_rt, 2):<7}| '
                  f'number of cars: {len(cars_routes):<2}| ATMs visited: {len(route):<3}')

    return car_routes_dict, cars_routes_times_dict


def split_route_by_cars(myroute: List[int],
                        myroute_time: List[int],
                        max_route_time: float=720.0) -> Tuple[Dict, Dict]:
    """Делим длинный маршрут между машинами

    Args:
        myroute (List[int]): _description_
        myroute_time (List[int]): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[Dict, Dict]: _description_
    """
    cars_routes = {}
    cars_routes_times = {}
    car_num = 1

    start = 0

    time_cumsum = 10
    i = 0

    while i < len(myroute_time):
        time_cumsum += myroute_time[i]

        if time_cumsum <= max_route_time:
            i += 1
            continue

        cars_routes[car_num] = deepcopy(myroute[start:i+1])
        cars_routes_times[car_num] = deepcopy(myroute_time[start:i])
        start = i + 1

        car_num += 1
        time_cumsum = 10
        i += 1

    cars_routes[car_num] = myroute[start:]

    if start != len(myroute_time):
        cars_routes_times[car_num] = myroute_time[start:]
    else:
        cars_routes_times[car_num] = [0]


    for _car, _times in cars_routes_times.items():
        cars_routes_times[_car][0] += 10 # Т.к. начальную точку мы инкассировали

        if sum(_times) > max_route_time:
            raise ValueError(f'Ошибка при разделении длинного маршрута: машина {_car}.'
                             f' {sum(_times)} > {max_route_time}')

        if len(cars_routes[_car]) - 1 != len(_times) and start != len(myroute_time):
            raise ValueError(f'Длина маршрута {len(cars_routes[_car])} не соответствует '
                             f'количеству временных отметок {len(_times)}: '
                             f'машина {_car}.')

    __full_rote = []
    for _car, _route in cars_routes.items():
        __full_rote += _route

    if __full_rote != myroute:
        raise ValueError('Изначальный и разделенный маршруты не совпадают')

    return cars_routes, cars_routes_times


def prepare_tsp_result(car_routes_dict: Dict[np.datetime64, Dict[int, List[int]]]) -> pd.DataFrame:
    """Из словаря с результатами оптимизации маршрутов инкассации формирует датафрейм

    Args:
        car_routes_dict (Dict[np.datetime64, Dict[int, List[int]]]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    results_list = []

    for date, cars_dict in car_routes_dict.items():
        for _car, _route in cars_dict.items():
            df = pd.DataFrame(columns=['TID', 'date', 'auto'])
            df['TID'] = _route
            df['auto'] = _car
            df['date'] = date
            df.index.name = 'number'
            df.reset_index(inplace=True)
            results_list.append(df.copy())

    return pd.concat(results_list, ignore_index=True)


if __name__ == '__main__':
    from datetime import datetime
    from bank_schedule.constants import RAW_DATA_FOLDER

    N_POINTS = 200

    myloader = Data(RAW_DATA_FOLDER)
    distances_df = myloader.get_distance_matrix().copy()
    tids_list = distances_df['Origin_tid'].sample(n=N_POINTS, random_state=0).tolist()

    start_t = datetime.now()
    res = solve_tsp_ortools(tids_list, distances_df, depot=tids_list[0], verbose=True)
    end_t = datetime.now()
    print(f'Time: {end_t - start_t}')
