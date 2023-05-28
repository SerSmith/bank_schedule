"""Ant colony optimization algorithm implementation
"""

import time
from typing import Any, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from bank_schedule.data import Data
from bank_schedule.constants import RAW_DATA_FOLDER

TIME_COL = 'Total_Time'
ORIG_COL = 'Origin_tid'
DEST_COL = 'Destination_tid'

class AntColonySolver:
    """Ant Colony Optimization Algorithm as applied to the Travelling Salesman Problem
    """
    def __init__(self,
                 distance_matrix: pd.DataFrame,
                 # run for a fixed amount of time
                 fix_time: int=0,
                 # minimum runtime
                 min_time: int=0,
                 # maximum time in seconds to run for
                 timeout: int=0,
                 # how many times to redouble effort after new new best path
                 stop_factor: int=2,
                 # minimum number of round trips before stopping
                 min_round_trips: int=10,
                 # maximum number of round trips before stopping
                 max_round_trips: int=0,
                 # Total number of ants to use
                 min_ants: int=0,
                 max_ants: int=0,
                 # this is the bottom of the near-optimal range for numpy performance
                 ant_count: int=64,
                 # how many steps do ants travel per epoch
                 ant_speed: int=1,
                 # power to which distance affects pheromones
                 distance_power=1,
                 # power to which differences in pheromones are noticed
                 pheromone_power=1.25,
                 # how fast do pheromones decay
                 decay_power=0,
                 # relative pheromone reward based on best_path_length/path_length
                 reward_power=0,
                 # queen multiplier for pheromones upon finding a new best path
                 best_path_smell=2,
                 # amount of starting pheromones [0 defaults to `10**self.distance_power`]
                 start_smell=0,
                 verbose=False,
    ):

        self.distance_matrix = distance_matrix
        self.fix_time = int(fix_time)
        self.min_time = int(min_time)
        self.timeout = int(timeout)
        self.stop_factor = float(stop_factor)
        self.min_round_trips = int(min_round_trips)
        self.max_round_trips = int(max_round_trips)
        self.min_ants = int(min_ants)
        self.max_ants = int(max_ants)

        self.ant_count = int(ant_count)
        self.ant_speed = int(ant_speed)

        self.distance_power = float(distance_power)
        self.pheromone_power = float(pheromone_power)
        self.decay_power = float(decay_power)
        self.reward_power = float(reward_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell = float(start_smell or 10**self.distance_power)

        self.verbose = int(verbose)
        self._initalized = False

        if self.min_round_trips and self.max_round_trips:
            self.min_round_trips = min(self.min_round_trips,
                                       self.max_round_trips)
        if self.min_ants and self.max_ants:
            self.min_ants = min(self.min_ants, self.max_ants)

        self.distances = distance_matrix
        self.distances.loc[self.distances[TIME_COL] == 0, TIME_COL] = 1

        self.pheromones = None
        self.ants_used = None
        self.epochs_used = None
        self.round_trips = None
        self.distance_cost = None

    def add_equals_to_distances(self):
        """_summary_
        """
        uniq_tids = self.distances[ORIG_COL].unique()
        # add rows for equal tids to pandas dataframe
        for tid in tqdm(uniq_tids):
            self.distances.loc[self.distances.shape[0], :] = [tid, tid, 0]

        self.distances[ORIG_COL] = self.distances[ORIG_COL].astype(int)
        self.distances[DEST_COL] = self.distances[DEST_COL].astype(int)


    def get_distance(self,
                     orig: int,
                     dest: int) -> float:
        """Get the pheromone value by node

        Args:
            this_node (int): _description_
            next_node (int): _description_
            source (pd.DataFrame): _description_

        Returns:
            float: _description_
        """
        cond = self.distances[ORIG_COL] == orig
        cond &= self.distances[DEST_COL] == dest

        return self.distances[cond][TIME_COL].values[0]


    def get_pair_index(self,
                       orig: int,
                       dest: int) -> int:
        """Get nodes pair index from distances matrix

        Args:
            this_node (int): _description_
            next_node (int): _description_
            source (pd.DataFrame): _description_

        Returns:
            float: _description_
        """
        cond = self.distances[ORIG_COL] == orig
        cond &= self.distances[DEST_COL] == dest

        return self.distances[cond].index[0]


    def solve_initialize(
            self,
            tids_list: List[Any]
    ) -> None:
        """Cache of distances between nodes

        Args:
            tids_list (List[Any]): _description_
        """

        # Cache of distance costs between nodes
        # division in a tight loop is expensive
        self.add_equals_to_distances()
        self.distance_cost = 1 / (1 + self.distances[TIME_COL]) ** self.distance_power

        ### This stores the pheromone trail that slowly builds up
        self.pheromones = pd.Series(data=self.start_smell, index=self.distances.index)

        ### Sanitise input parameters
        if self.ant_count <= 0:
            self.ant_count = len(tids_list)
        if self.ant_speed <= 0:
            self.ant_speed = self.distances[TIME_COL].median()
            self.ant_speed = self.ant_speed // 5

        self.ant_speed = int(max(1,self.ant_speed))

        ### Heuristic Exports
        self.ants_used   = 0
        self.epochs_used = 0
        self.round_trips = 0


    def solve(self,
              tids_list: List[Any],
              restart=False) -> List[Tuple[int,int]]:

        if restart or not self._initalized:
            self.solve_initialize(tids_list)

        ### Here come the ants!
        ants = {
            "distance": np.zeros((self.ant_count,)).astype('int32'),
            "path": [ [ tids_list[0] ] for _ in range(self.ant_count) ],
            "remaining": [ set(tids_list[1:]) for _ in range(self.ant_count) ],
            "path_cost": np.zeros((self.ant_count,)).astype('int32'),
            "round_trips": np.zeros((self.ant_count,)).astype('int32'),
        }

        best_path = None
        best_path_cost = np.inf
        best_epochs = []
        epoch = 0
        time_start = time.perf_counter()

        while True:
            epoch += 1

            ### Vectorized walking of ants
            # Small optimization here, testing against '> self.ant_speed' rather than '> 0'
            # avoids computing ants_arriving in the main part of this tight loop
            ants_travelling = ants['distance'] > self.ant_speed
            ants['distance'][ ants_travelling ] -= self.ant_speed
            if all(ants_travelling):
                # skip termination checks until the next ant arrives
                continue

            ### Vectorized checking of ants arriving
            ants_arriving = np.invert(ants_travelling)
            ants_arriving_index = np.where(ants_arriving)[0]

            for i in ants_arriving_index:

                ### ant has arrived at next_node
                this_node = ants['path'][i][-1]
                next_node = self.next_node(ants, i)

                ants['distance'][i]  = self.get_distance(this_node, next_node)
                ants['remaining'][i] = ants['remaining'][i] - {this_node}
                ants['path_cost'][i] = ants['path_cost'][i] + ants['distance'][i]
                ants['path'][i].append( next_node )

                ### ant has returned home to the colony
                if not ants['remaining'][i] and ants['path'][i][0] == ants['path'][i][-1]:
                    self.ants_used  += 1
                    self.round_trips = max(self.round_trips, ants["round_trips"][i] + 1)

                    ### We have found a new best path - inform the Queen
                    was_best_path = False
                    if ants['path_cost'][i] < best_path_cost:
                        was_best_path  = True
                        best_path_cost = ants['path_cost'][i]
                        best_path = ants['path'][i]
                        best_epochs += [ epoch ]
                        if self.verbose:
                            print({
                                "path_cost": int(ants['path_cost'][i]),
                                "ants_used": self.ants_used,
                                "epoch": epoch,
                                "round_trips": ants['round_trips'][i] + 1,
                                "clock": int(time.perf_counter() - time_start),
                            })

                    # leave pheromone trail
                    # doing this only after ants arrive home improves initial exploration
                    #  * self.round_trips has the effect of decaying old pheromone trails
                    # ** self.reward_power = -3 has the effect of encouraging ants
                    # to explore longer routes
                    # in combination with doubling pheromone for best_path
                    reward = 1

                    if self.reward_power:
                        reward *= ((best_path_cost / ants['path_cost'][i]) ** self.reward_power)
                    if self.decay_power:
                        reward *= (self.round_trips ** self.decay_power)

                    for path_index in range( len(ants['path'][i]) - 1 ):
                        this_node = ants['path'][i][path_index]
                        next_node = ants['path'][i][path_index+1]

                        this_next = self.get_pair_index(this_node, next_node)
                        next_this = self.get_pair_index(next_node, this_node)

                        self.pheromones[this_next] += reward
                        self.pheromones[next_this] += reward
                        if was_best_path:
                            # Queen orders to double the number of ants
                            # following this new best path
                            self.pheromones[this_next] *= self.best_path_smell
                            self.pheromones[next_this] *= self.best_path_smell

                    ### reset ant
                    ants["distance"][i] = 0
                    ants["path"][i] = [ tids_list[0] ]
                    ants["remaining"][i] = set(tids_list[1:])
                    ants["path_cost"][i] = 0
                    ants["round_trips"][i] += 1

            ### Do we terminate?

            # Always wait for at least 1 solutions (note: 2+ solutions are not guaranteed)
            if not len(best_epochs):
                continue

            # Timer takes priority over other constraints
            if self.fix_time or self.min_time or self.timeout:
                clock = time.perf_counter() - time_start
                if self.fix_time:
                    if clock > self.fix_time:
                        break
                    else:
                        continue
                if self.min_time and clock < self.min_time:
                    continue
                if self.timeout  and clock > self.timeout:
                    break

            # First epoch only has start smell - question:
            # how many epochs are required for a reasonable result?
            if self.min_round_trips and self.round_trips <  self.min_round_trips:
                continue
            if self.max_round_trips and self.round_trips >= self.max_round_trips:
                break

            # This factor is most closely tied to computational power   
            if self.min_ants and self.ants_used <  self.min_ants:
                continue
            if self.max_ants and self.ants_used >= self.max_ants:
                break

            # Lets keep redoubling our efforts until we can't find anything more
            if self.stop_factor and epoch > (best_epochs[-1] * self.stop_factor):
                break

        ### We have (hopefully) found a near-optimal path, report back to the Queen
        self.epochs_used = epoch
        self.round_trips = np.max(ants["round_trips"])
        return best_path


    def next_node(self, ants, index):
        this_node = ants['path'][index][-1]

        weights = []
        weights_sum = 0

        if not ants['remaining'][index]:
            # we will return home
            return ants['path'][index][0]

        for next_node in ants['remaining'][index]:

            if next_node == this_node:
                continue

            this_next = self.get_pair_index(this_node, next_node)

            reward = (
                    # Prefer shorter paths
                    self.pheromones[this_next] ** self.pheromone_power
                    * self.distance_cost[this_next]
            )
            weights.append( (reward, next_node) )
            weights_sum += reward

        # Pick a random path in proportion to the weight of the pheromone
        rand = np.random.random() * weights_sum
        for (weight, next_node) in weights:
            if rand <= weight:
                break
            rand -= weight

        return next_node


def split_route_sequence_into_pairs(tids_list: List[int]) -> List[List[int]]:
    """_summary_
    """
    return list(zip(tids_list, tids_list[1:]))


def get_cost_between_two_tids(start_tid: int,
                              end_tid: int,
                              distance_matrix: pd.DataFrame,
                              cost_col: str) -> float:
    """_summary_

    Args:
        start_tid (int): _description_
        end_tid (int): _description_
        distance_matrix (pd.DataFrame): _description_

    Returns:
        float: _description_
    """
    return distance_matrix[
        (distance_matrix[ORIG_COL]==start_tid)
        &
        (distance_matrix[DEST_COL]==end_tid)
        ][cost_col].iloc[0]



def get_route_time(tids_list: List[int],
                   distance_matrix: pd.DataFrame) -> float:
    """Считает время пути с посещением банкоматов из tids_list
    согласно их порядку в tids_list

    Args:
        tids_list (List[int]): порядок посещения банкоматов
        distance_matrix (pd.DataFrame): матрица расстояний с колонками
        [Origin_tid, Destination_tid, Total_Time]

    Returns:
        float: _description_
    """
    # split list into sublists of intersected tids pairs
    #  * [ (tid1, tid2), (tid2, tid3), (tid3, tid4), ... ]
    tids_pairs = split_route_sequence_into_pairs(tids_list)

    return sum(get_cost_between_two_tids(tid1,
                                         tid2,
                                         distance_matrix,
                                         TIME_COL) for tid1, tid2 in tids_pairs)


def run_aco_alorithm(tids_list: List[int],
                     verbose: bool=False,
                     label: Optional[Dict]=None,
                     **kwargs):
    """_summary_

    Args:
        tids_list (List[int]): _description_
        verbose (bool, optional): _description_. Defaults to False.
        plot (bool, optional): _description_. Defaults to False.
        label (dict, optional): _description_. Defaults to {}.
        algorithm (_type_, optional): _description_. Defaults to AntColonySolver.

    Returns:
        _type_: _description_
    """
    loader = Data(RAW_DATA_FOLDER)
    distance_matrix = loader.get_distance_matrix()

    solver = AntColonySolver(distance_matrix=distance_matrix,
                             verbose=verbose,
                             **kwargs)

    start_time = time.perf_counter()
    result = solver.solve(tids_list)
    stop_time = time.perf_counter()

    if label is not None:
        kwargs = { **label, **kwargs }

    for key in ['verbose', 'plot', 'animate', 'label', 'min_time', 'max_time']:
        if key in kwargs:
            del kwargs[key]

    n_tids = len(tids_list)

    random_route_time = get_route_time(np.random.permutation(tids_list),
                                       distance_matrix)
    result_route_time = get_route_time(result,
                                       distance_matrix)

    elapsed_time = stop_time - start_time

    print(f"n_tids={n_tids:<3d} | {random_route_time:5.0f} -> {result_route_time:4.0f} | "
          f"{elapsed_time:4.0f}s | ants: {solver.ants_used:5d} | trips: {solver.round_trips:4d} | "
          + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    )

    return result


if __name__ == '__main__':
    tids = [636538, 607066, 635440, 662968, 667509]
    loader = Data(RAW_DATA_FOLDER)
    distance_mx = loader.get_distance_matrix()
    print( get_route_time(tids, distance_mx) )

    run_aco_alorithm(tids, verbose=True, label={'algorithm': 'ACO'})
