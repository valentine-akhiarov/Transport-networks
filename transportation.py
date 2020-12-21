import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing.dummy import Pool as ThreadPool

# Turn interactive plotting off
plt.ioff()

from utilities import *

def calc_coords_in_circle_1(x, y, radius, x_size, y_size):
    """ Calculate coordinates in circle that satisfy radius & grid (must be located within) conditions

    :param x:      Current x coordinate in a grid
    :param y:      Current y coordinate in a grid
    :param radius: Desired radius of a circle
    :param x_size: Grid x size (of map to place people into)
    :param y_size: Grid y size (of map to place people into)
    :return:       Satisfying coordinates
    """

    coords = []

    # Possible y-shifts
    for i in range(-int(np.floor(radius)), int(np.floor(radius)) + 1):

        # Possible x-shifts
        for j in range(int(np.floor(radius)) + 1):

            # Coords in circle
            if np.sqrt(i ** 2 + j ** 2) <= radius:

                # Append valid coordinates
                if (x + j) >= 0 and (y + i) >= 0 and (x + j) < x_size and (y + i) < y_size:
                    coords.append((x + j, y + i))

                # Append valid mirrored coordinates
                if j > 0:

                    if (x - j) >= 0 and (y + i) >= 0 and (x - j) < x_size and (y + i) < y_size:
                        coords.append((x - j, y + i))

            # Coords not in circle
            else:

                break

    return np.array(coords)

def city_walk_iter(args):
    city, radius, neighbourhood_radius, observed_indexes, cur_x_arr, cur_y_arr, location_arr, home_x_arr, home_y_arr = args
    new_coords = np.empty(shape=(observed_indexes.shape[0], 2), dtype=np.int)

    for ii, i in enumerate(observed_indexes):
        x, y, location, x_home, y_home = cur_x_arr[ii], cur_y_arr[ii], location_arr[ii], home_x_arr[ii], home_y_arr[ii]
        # Skip working person
        if location != HOME:
            # print('skip_place')
            continue

        valid_coords = False

        # Get possible moves from current (x, y) location
        coords = calc_coords_in_circle_1(x, y, radius, city.x_size, city.y_size)

        # Randomly choose new move that yields neighbourhood coordinates
        c = 0
        while not valid_coords:
            c += 1
            if c > 100: 
                new_x, new_y = x_home, y_home
                break

            if coords.shape[0] == 0:
                new_x, new_y = x_home, y_home
                break

            rnd_move_idx = np.random.choice(coords.shape[0], 1, replace=False)[0]
            new_x, new_y = coords[rnd_move_idx]
            if np.sqrt(
                    (new_x - x_home) ** 2 + (new_y - y_home) ** 2) <= neighbourhood_radius:
                valid_coords = True

        new_coords[ii] = new_x, new_y

    return new_coords


def walk_iter(cities_list, radius, neighbourhood_radius, pool=None, num_threads=1):
    """ Walk people near their home

    :param cities_list:          List of CityResidents class objects
    :param radius:               Limiting radius for one epoch walk
    :param neighbourhood_radius: Maximum distance allowed to travel for each person from his home location
    """

    for city in cities_list:

        full_num = min(city.cur_x_arr.shape[0], city.cur_y_arr.shape[0], city.location_arr.shape[0], city.home_x_arr.shape[0], city.home_y_arr.shape[0])

        full_indexes = np.arange(full_num)
        step =  full_num // num_threads + 1

        args = [(city, radius, neighbourhood_radius, full_indexes[i:i+step], city.cur_x_arr[i:i+step], city.cur_y_arr[i:i+step], city.location_arr[i:i+step], city.home_x_arr[i:i+step], city.home_y_arr[i:i+step]) for i in np.arange(0, full_num, step)]

        a = [l for l in pool.map_async(city_walk_iter, args).get() if l.shape[0]]
        if a:
            a = np.concatenate(a)
            city.cur_x_arr, city.cur_y_arr = a[:,0], a[:,1]


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


if __name__=='__main__':
    print(':)')
