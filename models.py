import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from multiprocessing import get_context
# Turn interactive plotting off
plt.ioff()

from city_residents import *
from utilities import *
from transportation import *
from disease import *

def simulate_transportations_with_infections(init_transmitters_num, remote_workers, responsible_people,
                                             timer_min, timer_max, transmission_time, neighbourhood_radius,
                                             infect_prob, death_prob, radius, spread_radius,
                                             quarantine_zone_size,  transmitters_test_quota, others_test_quota,
                                             epochs, debug=False, plot_disease_matrix=None, num_threads=1, quarantine_start=39):
    """ Simulate people transportation and disease spread in a square grid

    :param init_transmitters_num:   Initial infected people number
    :param remote_workers:          Fraction of remote workers
    :param responsible_people:      Fraction of responsible people (which have lower probability of getting ill)
    :param timer_min:               Min steps (epochs) until infected person can transmit a disease (exception: initial group)
    :param timer_max:               Max steps (epochs) until infected person can transmit a disease (exception: initial group)
    :param transmission_time:       Disease lifespan
    :param neighbourhood_radius:    Maximum distance allowed to travel for each person from his initial location
    :param infect_prob:             Base probability for disease to transmit
    :param death_prob:              Death probability after disease
    :param radius:                  Maximum radius for person to travel in single epoch
    :param spread_radius:           Disease spreading radius
    :param quarantine_zone_size:    Quarantine zone's capacity
    :param transmitters_test_quota: Number of tests for visible transmitters to have possibility to move to quarantine zone with less death rate
    :param others_test_quota:       Number of tests for others to have possibility to move to quarantine zone with less death rate
    :param epochs:                  Steps to perform during each people 1) travel and 2) spread the disease
    :param debug:                   Debug mode to output functions run times
    :param plot_disease_matrix:     Path to save plot of disease (exposure) matrix before transmitting a disease in each epoch
    :return:                        Number of ill (visible + invisible) people for each epoch, number of ill (visible) people that can transmit a disease for each epoch
    """

    ###########################
    # Init variables          #
    ###########################

    cities_list = []

    msk = CityResidents(city_num=0, city_code='msk', x_size=506, y_size=506, residents_num=126781,
                        init_transmitters_num=init_transmitters_num,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    khi = CityResidents(city_num=1, city_code='khi', x_size=105, y_size=105, residents_num=2596,
                        init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    kra = CityResidents(city_num=2, city_code='kra', x_size=51, y_size=51, residents_num=1756, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    odi = CityResidents(city_num=3, city_code='odi', x_size=44, y_size=44, residents_num=1355, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    dom = CityResidents(city_num=4, city_code='dom', x_size=126, y_size=126, residents_num=1372,
                        init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    pod = CityResidents(city_num=5, city_code='pod', x_size=64, y_size=64, residents_num=3081, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    lub = CityResidents(city_num=6, city_code='lub', x_size=36, y_size=36, residents_num=2053, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    sho = CityResidents(city_num=7, city_code='sho', x_size=72, y_size=72, residents_num=1261, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    bal = CityResidents(city_num=8, city_code='bal', x_size=79, y_size=79, residents_num=5074, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    myt = CityResidents(city_num=9, city_code='myt', x_size=59, y_size=59, residents_num=2355, init_transmitters_num=0,
                        remote_workers=0, responsible_people=0, timer_min=timer_min,
                        timer_max=timer_max, transmission_time=transmission_time)

    cities_list.append(msk)
    cities_list.append(khi)
    cities_list.append(kra)
    cities_list.append(odi)
    cities_list.append(dom)
    cities_list.append(pod)
    cities_list.append(lub)
    cities_list.append(sho)
    cities_list.append(bal)
    cities_list.append(myt)

    # Add work location for msk residents
    msk.add_work_location(city_num=khi.city_num, x_size=khi.x_size, y_size=khi.y_size, workers_num=42)
    msk.add_work_location(city_num=kra.city_num, x_size=kra.x_size, y_size=kra.y_size, workers_num=29)
    msk.add_work_location(city_num=odi.city_num, x_size=odi.x_size, y_size=odi.y_size, workers_num=22)
    msk.add_work_location(city_num=dom.city_num, x_size=dom.x_size, y_size=dom.y_size, workers_num=22)
    msk.add_work_location(city_num=pod.city_num, x_size=pod.x_size, y_size=pod.y_size, workers_num=50)
    msk.add_work_location(city_num=lub.city_num, x_size=lub.x_size, y_size=lub.y_size, workers_num=33)
    msk.add_work_location(city_num=sho.city_num, x_size=sho.x_size, y_size=sho.y_size, workers_num=20)
    msk.add_work_location(city_num=bal.city_num, x_size=bal.x_size, y_size=bal.y_size, workers_num=82)
    msk.add_work_location(city_num=myt.city_num, x_size=myt.x_size, y_size=myt.y_size, workers_num=38)

    # Add work location for residents of other cities
    khi.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=695)
    kra.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=470)
    odi.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=363)
    dom.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=367)
    pod.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=825)
    lub.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=550)
    sho.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=338)
    bal.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=1359)
    myt.add_work_location(city_num=msk.city_num, x_size=msk.x_size, y_size=msk.y_size, workers_num=631)

    ###########################
    # Run simulations         #
    ###########################

    old_radius = radius
    old_neighbourhood_radius = neighbourhood_radius
    old_responsabl = responsible_people

    def gov_full_karatnine_start(epoch, radius, neighbourhood_radius, start_epoch=90, end_epoch=168):
        if epoch > start_epoch and epoch <= end_epoch:
            radius = 1
            neighbourhood_radius = 1
            for city in cities_list:
                city.init_remote_group(remote_workers)
        return radius, neighbourhood_radius

    def gov_karatnine_decrease(epoch, radius, neighbourhood_radius, start_epoch=168, end_epoch=450):
        if epoch <= end_epoch and epoch > start_epoch:
            radius = 1 + (old_radius-1) * (epoch-start_epoch) / (end_epoch - start_epoch)
            neighbourhood_radius = 1 + (old_neighbourhood_radius-2) * (epoch-start_epoch) / (end_epoch - start_epoch)
            for city in cities_list:
                # city.remote_workers = 
                city.init_remote_group(0.1 + (remote_workers-0.1) * (epoch-start_epoch) / (end_epoch - start_epoch))
        return radius, neighbourhood_radius

    def responsabl_fun(epoch):
        if epoch > 60:
            resp_probb = (1 - (1 - (transmitters_tracker[-1])*10/(126781))**200)**0.5
            for city in cities_list:
                city.init_responsible_group(old_responsabl*resp_probb) 


    timer_dict = defaultdict(list)

    healthy_tracker = []
    infected_tracker = []
    invisible_transmitters_tracker = []
    transmitters_tracker = []
    cured_tracker = []
    dead_tracker = []
    quarantine_tracker = []

    hour = 1
    pool = Pool(num_threads)
    for i in tqdm(range(1, epochs + 1)):
        radius, neighbourhood_radius = gov_full_karatnine_start(i, old_radius, old_neighbourhood_radius, start_epoch=quarantine_start, end_epoch=quarantine_start+120)
        radius, neighbourhood_radius = gov_karatnine_decrease(i, radius, neighbourhood_radius, start_epoch=quarantine_start+120, end_epoch=quarantine_start+360)
        responsabl_fun(i)
        if debug:
            print('radiuses', radius, neighbourhood_radius)
        a = cities_list[0].responsible_people_arr
        if debug:
            print('resp.sum', a[a == RESPONSIBLE].sum(-1))
        a = cities_list[0].worker_type_arr
        if debug:
            print('resp.sum', a[a == REMOTE].sum(-1))

        # 3 hours in a day
        if hour > 3:
            hour -= 3

        # Make disease visible (and transmittable) & finish disease
        if i > 1:
            start_time = time.time()
            decrement_timers(cities_list, transmission_time, death_prob)
            end_time = time.time()
            timer_dict['decrement_timers'].append(end_time - start_time)
            if debug:
                print('\tdecrement_timers()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Transport people to work
        elif hour == 2:
            start_time = time.time()
            transport_to_work(cities_list, amount='rest')
            end_time = time.time()
            timer_dict['transport_to_work'].append(end_time - start_time)
            if debug:
                print('\ttransport_to_work()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport one-third of the people to work
        #         if (hour % 7 == 0) & (hour % 2 != 0) & (hour % 3 != 0):
        #             start_time = time.time()
        #             transport_to_work(cities_list, amount='third')
        #             end_time = time.time()
        #             timer_dict['transport_to_work'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_work()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport one-third of the people to work
        #         elif (hour % 8 == 0) & (hour % 2 != 0) & (hour % 3 != 0):
        #             start_time = time.time()
        #             transport_to_work(cities_list, amount='third')
        #             end_time = time.time()
        #             timer_dict['transport_to_work'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_work()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport rest (one-third) of the people to work
        #         elif (hour % 9 == 0) & (hour % 2 != 0):
        #             start_time = time.time()
        #             transport_to_work(cities_list, amount='rest')
        #             end_time = time.time()
        #             timer_dict['transport_to_work'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_work()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Transport rest people from work
        elif hour == 3:
            start_time = time.time()
            transport_to_home(cities_list, amount='rest')
            end_time = time.time()
            timer_dict['transport_to_home'].append(end_time - start_time)
            if debug:
                print('\ttransport_to_home()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport one-third of the people from work
        #         elif hour % 19 == 0:
        #             start_time = time.time()
        #             transport_to_home(cities_list, amount='third')
        #             end_time = time.time()
        #             timer_dict['transport_to_home'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_home()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport one-third of the people from work
        #         elif hour % 20 == 0:
        #             start_time = time.time()
        #             transport_to_home(cities_list, amount='third')
        #             end_time = time.time()
        #             timer_dict['transport_to_home'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_home()\t\t{:.2f} sec.'.format(end_time - start_time))

        #         # Transport rest (one-third) of the people from work
        #         elif hour % 21 == 0:
        #             start_time = time.time()
        #             transport_to_home(cities_list, amount='rest')
        #             end_time = time.time()
        #             timer_dict['transport_to_home'].append(end_time - start_time)
        #             if debug:
        #                 print('\ttransport_to_home()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Walk peaple that are near their home
        start_time = time.time()
        walk_iter(cities_list, radius, neighbourhood_radius, pool, num_threads)
        end_time = time.time()
        # print(cities_list[0].cur_x_arr[:10])
        # print(cities_list[0].home_x_arr[:10])
        timer_dict['walk_iter'].append(end_time - start_time)
        if debug:
            print('\twalk_iter()\t\t\t{:.2f} sec.'.format(end_time - start_time))

        # Observe disease maps
        start_time = time.time()
        disease_mat_list = make_disease_matrices(cities_list, spread_radius)
        end_time = time.time()
        timer_dict['make_disease_matrices'].append(end_time - start_time)
        if debug:
            print('\tmake_disease_matrices()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Plot & save disease exposure map (cities)
        if plot_disease_matrix is not None:

            start_time = time.time()

            # Plot & save all cities on single figure
            # plot_disease_exposures(cities_list, spread_radius, epoch=i, path=plot_disease_matrix)

            # Plot & save all cities on multiple figures
            for city_idx in range(len(cities_list)):
                plot_disease_exposure(cities_list, city_idx, spread_radius, epoch=i, path=plot_disease_matrix)

            # Plot & save Moscow (Hub city)
            # plot_disease_exposure(cities_list, msk.city_num, spread_radius, epoch=i, path=plot_disease_matrix)

            end_time = time.time()
            timer_dict['plot_disease_exposure'].append(end_time - start_time)
            if debug:
                print('\tplot_disease_exposure()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Spread disease (based on the maps above)
        start_time = time.time()
        spread_disease(disease_mat_list, cities_list, timer_min, timer_max, transmission_time, infect_prob, pool, num_threads)
        end_time = time.time()
        timer_dict['spread_disease'].append(end_time - start_time)
        if debug:
            print('\tspread_disease()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Screen population to detect disease
        start_time = time.time()
        screen_for_disease(cities_list, quarantine_zone_size, transmitters_test_quota, others_test_quota)
        end_time = time.time()
        timer_dict['screen_for_disease'].append(end_time - start_time)
        if debug:
            print('\tscreen_for_disease()\t\t{:.2f} sec.'.format(end_time - start_time))

        # Track real stats
        start_time = time.time()
        track_stats(cities_list, healthy_tracker, infected_tracker, invisible_transmitters_tracker,
                    transmitters_tracker, cured_tracker, dead_tracker, quarantine_tracker)
        end_time = time.time()
        timer_dict['track_stats'].append(end_time - start_time)
        if debug:
            print('\ttrack_stats()\t\t\t{:.2f} sec.'.format(end_time - start_time))

        hour += 1

        # Debug
        print('[epoch={}]\tinfected={}\ttransmitters(visible+invisible)={}'.format(i, infected_tracker[-1],
                                                                                   invisible_transmitters_tracker[-1] +
                                                                                   transmitters_tracker[-1]))
        if debug:
            print('\n')
    pool.close()
    pool.join()
    # pool
    return timer_dict, \
           np.array(healthy_tracker), np.array(infected_tracker), np.array(invisible_transmitters_tracker), \
           np.array(transmitters_tracker), np.array(cured_tracker), np.array(dead_tracker), np.array(quarantine_tracker)