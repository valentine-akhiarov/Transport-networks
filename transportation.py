import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Turn interactive plotting off
plt.ioff()

from utilities import *

def walk_iter(cities_list, radius, neighbourhood_radius):
    """ Walk people near their home

    :param cities_list:          List of CityResidents class objects
    :param radius:               Limiting radius for one epoch walk
    :param neighbourhood_radius: Maximum distance allowed to travel for each person from his home location
    """

    for city in cities_list:

        for i, (x, y, location) in enumerate(zip(city.cur_x_arr, city.cur_y_arr, city.location_arr)):

            # Skip working person
            if location != HOME:
                continue

            valid_coords = False

            # Get possible moves from current (x, y) location
            coords = calc_coords_in_circle(x, y, radius, city.x_size, city.y_size)

            # Randomly choose new move that yields neighbourhood coordinates
            while not valid_coords:

                rnd_move_idx = np.random.choice(coords.shape[0], 1, replace=False)[0]
                new_x, new_y = coords[rnd_move_idx]
                if np.sqrt(
                        (new_x - city.home_x_arr[i]) ** 2 + (new_y - city.home_y_arr[i]) ** 2) <= neighbourhood_radius:
                    valid_coords = True

            city.cur_x_arr[i] = new_x
            city.cur_y_arr[i] = new_y


def transport_to_work(cities_list, amount='third'):
    """ Transport people to work across cities

    :param cities_list: List of CityResidents class objects
    """

    for city in cities_list:

        if amount == 'third':
            worker_indices = np.random.choice(
                np.where((city.location_arr == HOME) & (city.worker_type_arr == NORMAL))[0],
                int(np.floor((city.worker_type_arr == NORMAL).sum() / 3)), replace=False)

        elif amount == 'rest':
            worker_indices = np.where((city.location_arr == HOME) & (city.worker_type_arr == NORMAL))[0]

        else:
            print('[transport_to_work] amount value is undefined !')

        city.location_arr[worker_indices] = WORK


def transport_to_home(cities_list, amount='third'):
    """ Transport people to home across cities

    :param cities_list: List of CityResidents class objects
    """

    for city in cities_list:

        if amount == 'third':
            worker_indices = np.random.choice(
                np.where((city.location_arr == WORK) & (city.worker_type_arr == NORMAL))[0],
                int(np.floor((city.worker_type_arr == NORMAL).sum() / 3)), replace=False)

        elif amount == 'rest':
            worker_indices = np.where((city.location_arr == WORK) & (city.worker_type_arr == NORMAL))[0]

        else:
            print('[transport_to_home] amount value is undefined !')

        city.location_arr[worker_indices] = HOME

        # Reset coordinates to home
        city.cur_x_arr[worker_indices] = city.home_x_arr[worker_indices]
        city.cur_y_arr[worker_indices] = city.home_y_arr[worker_indices]