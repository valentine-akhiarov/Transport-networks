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

class CityResidents:

    def __init__(self, city_num, city_code, x_size, y_size, residents_num, init_transmitters_num, remote_workers,
                 responsible_people, timer_min, timer_max, transmission_time):
        self.city_num = city_num
        self.city_code = city_code
        self.x_size = x_size
        self.y_size = y_size
        self.residents_num = residents_num

        self.init_home_coords()
        self.init_current_location()
        self.init_work_location()
        self.init_disease_arrays()
        self.init_infected_group(init_transmitters_num, transmission_time)
        self.init_remote_group(remote_workers)
        self.init_responsible_group(responsible_people)

    def init_home_coords(self):
        self.home_x_arr = np.random.randint(self.x_size, size=self.residents_num)
        self.home_y_arr = np.random.randint(self.y_size, size=self.residents_num)

    def init_current_location(self):
        self.cur_x_arr = self.home_x_arr.copy()
        self.cur_y_arr = self.home_y_arr.copy()
        self.location_arr = np.zeros(self.residents_num).astype(int)

    def init_disease_arrays(self):
        self.status_arr = np.zeros(self.residents_num).astype(int)  # Illness status
        self.incubation_timer_arr = np.zeros(self.residents_num).astype(
            int)  # Timer untill illness becomes observable (& transmittable)
        self.transmission_timer_arr = np.zeros(self.residents_num).astype(int)  # Timer untill illness vanishes

    def init_infected_group(self, init_transmitters_num, transmission_time):
        doomed_indices = np.random.choice(self.residents_num, init_transmitters_num, replace=False)
        self.status_arr[doomed_indices] = INVISIBLE_TRANSMITTER
        self.transmission_timer_arr[doomed_indices] = transmission_time

    def init_work_location(self):
        # Init working place as a city of residence
        self.work_city_arr = np.array(self.residents_num * [self.city_num])
        self.work_x_arr = np.random.randint(self.x_size, size=self.residents_num)
        self.work_y_arr = np.random.randint(self.y_size, size=self.residents_num)

    def init_remote_group(self, remote_workers):
        self.worker_type_arr = np.zeros(self.residents_num).astype(int)
        self.worker_type_arr[:] = NORMAL

        # Randomly choose remote workers
        remote_worker_indices = np.random.choice(self.residents_num,
                                                 size=int(np.floor(remote_workers * self.residents_num)), replace=False)
        self.worker_type_arr[remote_worker_indices] = REMOTE

    def init_responsible_group(self, responsible_people):
        self.responsible_people_arr = np.zeros(self.residents_num).astype(int)
        self.responsible_people_arr[:] = RECKLESS

        # Randomly choose responsible people
        responsible_people_indices = np.random.choice(self.residents_num,
                                                      size=int(np.floor(responsible_people * self.residents_num)),
                                                      replace=False)
        self.responsible_people_arr[responsible_people_indices] = RESPONSIBLE

    def add_work_location(self, city_num, x_size, y_size, workers_num):
        # Randomly select 'workers_num' who will work in other city
        indices = np.random.choice(np.where(self.work_city_arr == self.city_num)[0], workers_num, replace=False)
        self.work_x_arr[indices] = np.random.randint(x_size, size=workers_num)
        self.work_y_arr[indices] = np.random.randint(y_size, size=workers_num)
        self.work_city_arr[indices] = city_num