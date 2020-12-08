import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Turn interactive plotting off
plt.ioff()

# Location
HOME = 0
WORK = 1
QUARANTINE = 2

# Responsibility
RECKLESS = 0
RESPONSIBLE = 1

# Worker type
NORMAL = 0
REMOTE = 1

# Status
HEALTHY = 0
INFECTED = 1
INVISIBLE_TRANSMITTER = 2
TRANSMITTER = 3
CURED = 4
DEAD = 5

def calc_coords_in_circle(x, y, radius, x_size, y_size):
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


def decrement_timers(cities_list, transmission_time, death_prob):
    """ Decrement incubation & transmission timers across cities

    :param cities_list:       List of CityResidents class objects
    :param transmission_time: Disease lifespan
    :param death_prob:        Probability of death after disease
    """

    for city in cities_list:
        # Make illness observable
        new_disease_observations = np.where(city.incubation_timer_arr == 1)[0]
        city.status_arr[new_disease_observations] = INVISIBLE_TRANSMITTER

        # Decrement people counter with unobservable illness
        disease_indices = np.where(city.incubation_timer_arr > 0)[0]
        city.incubation_timer_arr[disease_indices] -= 1

        # Finish illness - choose outcome
        disease_finish_observations = np.where((city.transmission_timer_arr == 1) & (city.location_arr != QUARANTINE))[
            0]
        city.status_arr[disease_finish_observations] = np.random.choice([CURED, DEAD],
                                                                        size=disease_finish_observations.size,
                                                                        p=[1 - death_prob, death_prob], replace=True)

        disease_finish_observations = np.where((city.transmission_timer_arr == 1) & (city.location_arr == QUARANTINE))[
            0]
        city.status_arr[disease_finish_observations] = np.random.choice([CURED, DEAD],
                                                                        size=disease_finish_observations.size,
                                                                        p=[1 - death_prob / 4, death_prob / 4],
                                                                        replace=True)
        city.location_arr[disease_finish_observations] = HOME

        # Decrement people counter with observable & transmittable illness
        disease_indices = np.where(city.transmission_timer_arr > 0)[0]
        city.transmission_timer_arr[disease_indices] -= 1

        # Make transmitters visible
        disease_indices = np.where(
            (city.transmission_timer_arr <= int(np.floor(transmission_time / 2))) & (city.transmission_timer_arr != 0))[
            0]
        city.status_arr[disease_indices] = TRANSMITTER

        # Set timer for new transmitters
        city.transmission_timer_arr[new_disease_observations] = transmission_time


def track_stats(cities_list, healthy_tracker, infected_tracker, invisible_transmitters_tracker, transmitters_tracker,
                cured_tracker, dead_tracker, quarantine_tracker):
    """ Track infected & transmitters

    :param cities_list:                     List of CityResidents class objects
    :param healthy_tracker:                 List of healthy people for epochs
    :param infected_tracker:                List of infected people for epochs
    :param invisible_transmitters_tracker:  List of invisible trasmitters for epochs
    :param transmitters_tracker:            List of visible trasmitters for epochs
    :param cured_tracker:                   List of cured people for epochs
    :param dead_tracker:                    List of dead people for epochs
    :param quarantine_tracker:              List of people in quarantine
    """

    healthy = 0
    infected = 0
    invisible_transmitters = 0
    transmitters = 0
    cured = 0
    dead = 0
    quarantined = 0

    for city in cities_list:
        healthy += (city.status_arr == HEALTHY).sum()
        infected += (city.status_arr == INFECTED).sum()
        invisible_transmitters += (city.status_arr == INVISIBLE_TRANSMITTER).sum()
        transmitters += (city.status_arr == TRANSMITTER).sum()
        cured += (city.status_arr == CURED).sum()
        dead += (city.status_arr == DEAD).sum()
        quarantined += (city.location_arr == QUARANTINE).sum()

    healthy_tracker.append(healthy)
    infected_tracker.append(infected)
    invisible_transmitters_tracker.append(invisible_transmitters)
    transmitters_tracker.append(transmitters)
    cured_tracker.append(cured)
    dead_tracker.append(dead)
    quarantine_tracker.append(quarantined)