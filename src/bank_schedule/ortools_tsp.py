"""Simple Travelling Salesperson Problem (TSP) between cities."""

from typing import List, Tuple

import pandas as pd

from bank_schedule.data import distances_matrix_from_dataframe

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


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

    if solution:
        steps, steps_times, summary_time = get_route_and_time(manager, routing, solution)
        steps = [tids_list[i] for i in steps]
        return steps[:-1], steps_times[:-1], summary_time

    return [], [], -1


if __name__ == '__main__':
    from bank_schedule.data import Data
    from bank_schedule.constants import RAW_DATA_FOLDER

    N_POINTS = 30

    loader = Data(RAW_DATA_FOLDER)
    distances_df = loader.get_distance_matrix()
    tids_list = distances_df['Origin_tid'].sample(n=N_POINTS, random_state=0).tolist()

    print(tids_list)
    print( solve_tsp_ortools(tids_list, distances_df, depot=tids_list[0]) )
