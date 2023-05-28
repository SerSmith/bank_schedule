"""Simple Travelling Salesperson Problem (TSP) between cities."""

from typing import List, Tuple

import pandas as pd
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from bank_schedule.data import distances_matrix_from_dataframe
from bank_schedule import scheduler
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
    distances_df = loader.get_distance_matrix()

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
                    residuals: pd.DataFrame,
                    atms_per_day: int=200,
                    mandatory_selection_threshold: int=1,
                    n_iterations: int=1,
                    neighborhood_radius: float=15,
                    max_route_time: float=720.0,
                    mandatory_selection_col: str='days_to_deadline',
                    tids_col: str='TID',
                    use_greedy: bool=True):
    """Находит количество машин, необходимое для обслуживания банкоматов
    в день

    Args:
        loader (Data): класс загрузки данных
        residuals (pd.DataFrame): данные об остатках, подготовленные для составления расписания
        (см. scheduler.prepare_residual_to_schedule_creation)
        atms_per_day (int, optional): _description_. Defaults to 150.
        mandatory_selection_threshold (int, optional): _description_. Defaults to 1.
        n_iterations (int, optional): _description_. Defaults to 1.
        neighborhood_radius (float, optional): _description_. Defaults to 15.
        max_route_time (float, optional): _description_. Defaults to 720.0.
        mandatory_selection_col (str, optional): _description_. Defaults to 'days_to_deadline'.
        tid_col (str, optional): _description_. Defaults to 'TID'.

    Returns:
        _type_: _description_
    """

    atms_df = scheduler.get_atms_for_today_collection(loader,
                                                      residuals,
                                                      atms_per_day,
                                                      mandatory_selection_threshold,
                                                      neighborhood_radius,
                                                      mandatory_selection_col=mandatory_selection_col,
                                                      tids_col=tids_col,
                                                      use_greedy=use_greedy)
    tids_for_collection = atms_df[tids_col].tolist()

    route, route_time, route_time_sum = get_best_route(loader,
                                                        tids_for_collection,
                                                        n_iterations=n_iterations)

    return route, route_time, route_time_sum / max_route_time


if __name__ == '__main__':
    from datetime import datetime
    from bank_schedule.data import Data
    from bank_schedule.constants import RAW_DATA_FOLDER

    N_POINTS = 200

    myloader = Data(RAW_DATA_FOLDER)
    distances_df = myloader.get_distance_matrix()
    tids_list = distances_df['Origin_tid'].sample(n=N_POINTS, random_state=0).tolist()

    start = datetime.now()
    res = solve_tsp_ortools(tids_list, distances_df, depot=tids_list[0], verbose=True)
    end = datetime.now()
    print(f'Time: {end - start}')
