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


def make_disease_matrix(cities_list, city_idx, spread_radius):
    """ Make disease (exposure) matrix which indicates for each spot a number of ill people near, that can transfer a disease

    :param cities_list:   List of CityResidents class objects
    :param city_idx:      Index of city of interest in cities_list
    :param spread_radius: Radius of exposure of an infected person within which disease can be transfered to a healthy one
    :return:              Disease (exposure) matrix
    """

    city = cities_list[city_idx]

    # Init disease matrix
    disease_mat = np.zeros(shape=(city.y_size, city.x_size)).astype(int)

    # Get indices of residents at home with trasmittable disease
    disease_indices = np.where(
        (city.location_arr == HOME) & ((city.status_arr == TRANSMITTER) | (city.status_arr == INVISIBLE_TRANSMITTER)))[
        0]

    # Calculate disease spread for each infected person
    for i in disease_indices:

        infected_coords = calc_coords_in_circle(city.cur_x_arr[i], city.cur_y_arr[i], spread_radius, city.x_size,
                                                city.y_size)

        for x, y in infected_coords:
            disease_mat[y, x] += 1

    # Process workers (residents + from other cities) with trasmittable disease
    for n in range(len(cities_list)):

        # Get indices of workers from other cities with trasmittable disease
        disease_indices = np.where(
            (cities_list[n].work_city_arr == city_idx) & (cities_list[n].location_arr == WORK) & (
                        (cities_list[n].status_arr == TRANSMITTER) | (
                            cities_list[n].status_arr == INVISIBLE_TRANSMITTER)))[0]

        # Calculate disease spread for each infected person
        for i in disease_indices:

            infected_coords = calc_coords_in_circle(cities_list[n].work_x_arr[i], cities_list[n].work_y_arr[i],
                                                    spread_radius, city.x_size, city.y_size)

            for x, y in infected_coords:
                disease_mat[y, x] += 1

    return disease_mat


def make_disease_matrices(cities_list, spread_radius):
    """ Make disease (exposure) matrices for each city

    :param cities_list:   List of CityResidents class objects
    :param spread_radius: Radius of exposure of an infected person within which disease can be transfered to a healthy one
    :return:              List of disease (exposure) matrices
    """

    disease_mat_list = []

    for city_idx in range(len(cities_list)):
        disease_mat = make_disease_matrix(cities_list, city_idx, spread_radius)
        disease_mat_list.append(disease_mat)

    return disease_mat_list


def calc_infection_prob(infection_exposure, infect_prob=0.5):
    """ Calculate infection probability based on exposure and infection's base probability

    :param infection_exposure: Number of infected people which can transfer the disease and located near a healthy person
    :param infect_prob:        Infection's base probability
    :return:                   Probability of infection
    """
    return 1 - (1 - infect_prob) ** infection_exposure


def spread_disease(disease_mat_list, cities_list, timer_min, timer_max, transmission_time, infect_prob):
    """ Spread disease between infected and healthy people based on disease_mat_list. Updates status_arr & incubation_timer_arr in cities_list objects

    :param disease_mat_list:  List of maps of overlapping areas of disease exposure from current infected people
    :param cities_list:       List of CityResidents class objects
    :param timer_min:         Min steps (epochs) until infected person can transmit a disease (exception: initial group)
    :param timer_max:         Max steps (epochs) until infected person can transmit a disease (exception: initial group)
    :param transmission_time: Disease lifespan
    :param infect_prob:       Disease's base probability of transmission
    """

    for city_idx, (city, disease_mat) in enumerate(zip(cities_list, disease_mat_list)):

        # Get indices of healthy residents staying at home
        disease_indices = np.where((city.location_arr == HOME) & (city.status_arr == HEALTHY))[0]

        # Transmit disease between residents in city
        for i in disease_indices:

            infection_exposure = disease_mat[city.cur_y_arr[i], city.cur_x_arr[i]]

            # Calculate probability of getting ill and illness' outcome
            infection_prob = calc_infection_prob(infection_exposure, infect_prob)
            if city.responsible_people_arr[i] == RESPONSIBLE:
                infection_prob /= 10
            infection_outcome = np.random.choice([HEALTHY, INFECTED], p=[1 - infection_prob, infection_prob],
                                                 replace=False)

            # Infect only healthy people
            assert city.status_arr[i] == HEALTHY
            if infection_outcome == INFECTED and city.incubation_timer_arr[i] == 0:
                city.status_arr[i] = infection_outcome
                # city.incubation_timer_arr[i] = np.random.choice(range(timer_min, timer_max + 1), replace=False)
                city.incubation_timer_arr[i] = timer_max

                # Instant transmitter
                if city.incubation_timer_arr[i] == 0:
                    city.status_arr[i] = INVISIBLE_TRANSMITTER
                    city.transmission_timer_arr[i] = transmission_time

        # Transmit disease between working residents + other workers in city
        for n in range(len(cities_list)):

            # Get indices of healthy workers from other cities
            disease_indices = np.where(
                (cities_list[n].work_city_arr == city_idx) & (cities_list[n].location_arr == WORK) & (
                            cities_list[n].status_arr == HEALTHY))[0]

            # Transmit disease between other workers in city
            for i in disease_indices:

                infection_exposure = disease_mat[cities_list[n].work_y_arr[i], cities_list[n].work_x_arr[i]]

                # Calculate probability of getting ill and illness' outcome
                infection_prob = calc_infection_prob(infection_exposure, infect_prob)
                if cities_list[n].responsible_people_arr[i] == RESPONSIBLE:
                    infection_prob /= 10
                infection_outcome = np.random.choice([HEALTHY, INFECTED], p=[1 - infection_prob, infection_prob],
                                                     replace=False)

                # Infect only healthy people
                assert cities_list[n].status_arr[i] == HEALTHY
                if infection_outcome == INFECTED and cities_list[n].incubation_timer_arr[i] == 0:
                    cities_list[n].status_arr[i] = infection_outcome
                    # cities_list[n].incubation_timer_arr[i] = np.random.choice(range(timer_min, timer_max + 1), replace=False)
                    cities_list[n].incubation_timer_arr[i] = timer_max

                    # Instant transmitter
                    if cities_list[n].incubation_timer_arr[i] == 0:
                        cities_list[n].status_arr[i] = INVISIBLE_TRANSMITTER
                        cities_list[n].transmission_timer_arr[i] = transmission_time


def plot_disease_exposure(cities_list, city_idx, spread_radius, epoch=None, path=None, fig=None, ax=None):
    """ Plot disease (exposure) matrix for one city

    :param cities_list:   List of CityResidents class objects
    :param city_idx:      Index of city of interest in cities_list
    :param spread_radius: Disease spreading radius
    :param epoch:         Current epoch in simulation
    :param path:          Absolute path to save plot to
    :param fig:           Figure object
    :param ax:            Axis to plot
    """

    city = cities_list[city_idx]

    state = 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        state = 1

    ax.set_xlim(0, city.x_size - 1)
    ax.set_ylim(0, city.y_size - 1)

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi

    # Plot all residents that are in that city
    home_indices = np.where(city.location_arr == HOME)[0]
    ax.scatter(city.cur_x_arr[home_indices], city.cur_y_arr[home_indices], s=1, c='grey', alpha=0.1, label='Uninfected')

    # Plot all other workers (residents + aliens) in that city
    for n in range(len(cities_list)):
        work_indices = np.where((cities_list[n].work_city_arr == city_idx) & (cities_list[n].location_arr == WORK))[0]
        ax.scatter(cities_list[n].work_x_arr[work_indices], cities_list[n].work_y_arr[work_indices], s=1, c='grey',
                   alpha=0.1)

    # Plot all infected residents at home that are in that city
    ax.scatter(city.work_x_arr[(city.status_arr == INFECTED) & (city.location_arr == HOME)],
               city.work_y_arr[(city.status_arr == INFECTED) & (city.location_arr == HOME)],
               s=(spread_radius * np.mean([width]) / city.x_size), c='yellow', alpha=0.5, label='Infected')

    # Plot all other infected workers (residents + aliens) in that city
    for n in range(len(cities_list)):
        work_indices = np.where((cities_list[n].work_city_arr == city_idx) & (cities_list[n].location_arr == WORK) & (
                    cities_list[n].status_arr == INFECTED))[0]
        ax.scatter(cities_list[n].work_x_arr[work_indices], cities_list[n].work_y_arr[work_indices],
                   s=(spread_radius * np.mean([width]) / city.x_size), c='yellow', alpha=0.5)

    # Plot all residents-transmitters at home that are in that city
    ax.scatter(city.work_x_arr[((city.status_arr == TRANSMITTER) | (city.status_arr == INVISIBLE_TRANSMITTER)) & (
                city.location_arr == HOME)],
               city.work_y_arr[((city.status_arr == TRANSMITTER) | (city.status_arr == INVISIBLE_TRANSMITTER)) & (
                           city.location_arr == HOME)], s=(spread_radius * np.mean([width]) / city.x_size), c='red',
               alpha=1, label='Transmitters (visible + invisible)')

    # Plot all other workers-transmitters (residents + aliens) in that city
    for n in range(len(cities_list)):
        work_indices = np.where((cities_list[n].work_city_arr == city_idx) & (cities_list[n].location_arr == WORK) & (
                    (cities_list[n].status_arr == TRANSMITTER) | (cities_list[n].status_arr == INVISIBLE_TRANSMITTER)))[0]
        ax.scatter(cities_list[n].work_x_arr[work_indices], cities_list[n].work_y_arr[work_indices],
                   s=(spread_radius * np.mean([width]) / city.x_size), c='red', alpha=1)

    ax.set_title(f'City {city_idx}={city.city_code}. Disease (exposure) matrix - transmitters (in a city) = '
                 f'{((city.status_arr == TRANSMITTER) | (city.status_arr == INVISIBLE_TRANSMITTER)).sum()}, '
                 f'infected = {(city.status_arr == INFECTED).sum()} - epoch {epoch}')
    ax.legend(loc=2)

    if state == 1:

        if epoch is not None and path is not None:

            # Create folder if it doesn't exist
            if not os.path.exists(path):
                os.mkdir(path)

            # Save plot
            fig.savefig(os.path.join(path, f'City_{city_idx}_{city.city_code}_disease_matrix_epoch_{epoch}.png'),
                        dpi=300)

        plt.close(fig)


def plot_disease_exposures(cities_list, spread_radius, epoch=None, path=None):
    """ Plot disease (exposure) matrix for one city

    :param cities_list:   List of CityResidents class objects
    :param spread_radius: Disease spreading radius
    :param epoch:         Current epoch in simulation
    :param path:          Absolute path to save plot to
    """

    plot_dict = {
        0: {'x': 1, 'y': 1},
        1: {'x': 1, 'y': 0},
        2: {'x': 0, 'y': 1},
        3: {'x': 0, 'y': 2},
        4: {'x': 2, 'y': 3},
        5: {'x': 1, 'y': 3},
        6: {'x': 3, 'y': 2},
        7: {'x': 3, 'y': 1},
        8: {'x': 3, 'y': 0},
        9: {'x': 2, 'y': 0}
    }

    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(4 * 15, 4 * 15))

    for city_idx in range(len(cities_list)):
        plot_disease_exposure(cities_list, city_idx, spread_radius, epoch=epoch, path=path,
                              fig=fig, ax=ax[plot_dict[city_idx]['y'], plot_dict[city_idx]['x']])

    if epoch is not None and path is not None:

        # Create folder if it doesn't exist
        if not os.path.exists(path):
            os.mkdir(path)

        # Save plot
        fig.savefig(os.path.join(path, f'All_cities_disease_matrix_epoch_{epoch}.png'), dpi=300)

    plt.close(fig)


def _screen_transmitters(cities_list, city_idx, quarantine_zone_size, quarantime_occupancy, transmitter_candidates, transmitters_test_quota):
    """ Screen transmitters from single city for disease to move some into quarantine zone with less death rate

    :param cities_list:             List of CityResidents class objects
    :param city_idx:                City index
    :param quarantine_zone_size:    Quarantine zone's size
    :param quarantime_occupancy:    Quarantine zone's current occupancy
    :param transmitter_candidates:  Total test candidates
    :param transmitters_test_quota: Number of tests for visible transmitters
    :return: Current quarantine zone's current occupancy
    """

    city = cities_list[city_idx]

    if transmitter_candidates == 0:
        return quarantime_occupancy

    # candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr != DEAD) & (city.status_arr != CURED)
    candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr == TRANSMITTER)
    city_candidates = candidate_mask.sum()
    city_ratio = city_candidates / transmitter_candidates  # total_candidates
    city_quota = int(np.floor(city_ratio * transmitters_test_quota))  # Approximation - rounding down

    # Screen randomly selected transmitters
    candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr == TRANSMITTER)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        return quarantime_occupancy

    # Quarantine zone is not full yet
    if quarantime_occupancy < quarantine_zone_size:

        # Fill last spaces
        if quarantine_zone_size - quarantime_occupancy < city_quota:
            candidate_indices = np.random.choice(candidate_indices, min(quarantine_zone_size - quarantime_occupancy,
                                                                        candidate_indices.size), replace=False)
            city.location_arr[candidate_indices] = QUARANTINE
            quarantime_occupancy += candidate_indices.size
            return quarantime_occupancy

        # Transfer all transmitters to quarantine zone
        else:
            candidate_indices = np.random.choice(candidate_indices, min(city_quota, candidate_indices.size),
                                                 replace=False)
            city.location_arr[candidate_indices] = QUARANTINE
            quarantime_occupancy += city_quota
            return quarantime_occupancy


def _screen_others(cities_list, city_idx, quarantine_zone_size, quarantime_occupancy, other_candidates, others_test_quota):
    """ Screen others (healthy + infected) from single city for disease to move some into quarantine zone with less death rate

    :param cities_list:             List of CityResidents class objects
    :param city_idx:                City index
    :param quarantine_zone_size:    Quarantine zone's size
    :param quarantime_occupancy:    Quarantine zone's current occupancy
    :param other_candidates:        Total test candidates
    :param others_test_quota:       Number of tests for healthy/infected
    :return: Current quarantine zone's current occupancy
    """

    city = cities_list[city_idx]

    if other_candidates == 0:
        return quarantime_occupancy

    # candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr != DEAD) & (city.status_arr != CURED)
    candidate_mask = (city.location_arr != QUARANTINE) & ((city.status_arr == HEALTHY) | (city.status_arr == INFECTED))
    city_candidates = candidate_mask.sum()
    city_ratio = city_candidates / other_candidates  # total_candidates
    city_quota = int(np.floor(city_ratio * others_test_quota))  # Approximation - rounding down

    # Screen randomly selected others
    infected_mask = (city.location_arr != QUARANTINE) & (city.status_arr == INFECTED)
    healthy_mask = (city.location_arr != QUARANTINE) & (city.status_arr == HEALTHY)
    if infected_mask.sum() == 0:
        return quarantime_occupancy

    # candidate_mask = (city.location_arr != QUARANTINE) & ((city.status_arr == INFECTED) | (city.status_arr == HEALTHY))
    # candidate_indices = np.where(candidate_mask)[0]

    # Simulate random selection of infected group from infected + healthy population
    # if city_quota >= infected_mask.sum() + healthy_mask.sum():

    if (infected_mask.sum() + healthy_mask.sum()) == 0:
        return quarantime_occupancy

    # Enough quota to quarantine all infected
    if city_quota / (infected_mask.sum() + healthy_mask.sum()) >= 1:
        candidate_indices = np.where(infected_mask)[0]

    # Select those who are infected and quarantine 'em
    else:
        candidate_indices = np.append(np.where(infected_mask)[0], np.where(healthy_mask)[0])

        # Array to show if infected person is going through screening
        selected_infected_mask = np.random.choice([True, False], infected_mask.sum(),
                                                  p=[city_quota / candidate_indices.size,
                                                     1 - city_quota / candidate_indices.size])
        candidate_indices = np.where(infected_mask)[0][selected_infected_mask]

        # candidate_indices = np.random.choice(candidate_indices, int(
        #     np.ceil(infected_mask.sum() * (city_quota / (infected_mask.sum() + healthy_mask.sum())))),
        #                                      replace=False)

    if candidate_indices.size == 0:
        return quarantime_occupancy

    # Quarantine zone is not full yet
    if quarantime_occupancy < quarantine_zone_size:

        # Fill last spaces
        if quarantine_zone_size - quarantime_occupancy < city_quota:
            candidate_indices = np.random.choice(candidate_indices, min(quarantine_zone_size - quarantime_occupancy,
                                                                        candidate_indices.size), replace=False)
            city.location_arr[candidate_indices] = QUARANTINE
            quarantime_occupancy += candidate_indices.size
            return quarantime_occupancy

        # Transfer all transmitters to quarantine zone
        else:
            candidate_indices = np.random.choice(candidate_indices, min(city_quota, candidate_indices.size),
                                                 replace=False)
            city.location_arr[candidate_indices] = QUARANTINE
            quarantime_occupancy += city_quota
            return quarantime_occupancy


def screen_for_disease(cities_list, quarantine_zone_size, transmitters_test_quota, others_test_quota):
    """ Screen population for disease to move some into quarantine zone with less death rate

    :param cities_list:             List of CityResidents class objects
    :param quarantine_zone_size:    Quarantine zone's capacity
    :param transmitters_test_quota: Number of tests for visible transmitters
    :param others_test_quota:       Number of tests for others
    """

    # Calc candidates for screening in all cities
    total_candidates = 0
    transmitter_candidates = 0
    other_candidates = 0
    quarantime_occupancy = 0

    for city in cities_list:
        candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr != DEAD) & (city.status_arr != CURED)
        total_candidates += candidate_mask.sum()

        candidate_mask = (city.location_arr != QUARANTINE) & (city.status_arr == TRANSMITTER)
        transmitter_candidates += candidate_mask.sum()

        candidate_mask = (city.location_arr != QUARANTINE) & ((city.status_arr == HEALTHY) | (city.status_arr == INFECTED))
        other_candidates += candidate_mask.sum()

    quarantime_occupancy += (city.location_arr == QUARANTINE).sum()

    # Screen cities in random order by transmitters/others groups
    priority_queue = list((i, 'transmitters') for i in range(len(cities_list))) + list((i, 'others') for i in range(len(cities_list)))
    np.random.shuffle(priority_queue)

    for city_idx, group in priority_queue:

        # Screen other group
        if group == 'others':

            quarantime_occupancy = _screen_others(cities_list, city_idx, quarantine_zone_size, quarantime_occupancy,
                                                  other_candidates, others_test_quota)

        # Screen transmitter group
        elif group == 'transmitters':

            quarantime_occupancy = _screen_transmitters(cities_list, city_idx,
                                                        quarantine_zone_size, quarantime_occupancy,
                                                        transmitter_candidates, transmitters_test_quota)

