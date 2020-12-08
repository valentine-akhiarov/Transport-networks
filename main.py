

DEBUG = True                    # Output each main function runtime

# 1 person = 100 people
init_transmitters_num = 1       # Initial infected people number

timer_min = 0                   # Min steps (epochs) until infected person can transmit a disease (exception: initial group)
timer_max = 21                  # Max steps (epochs) until infected person can transmit a disease (exception: initial group)
transmission_time = 42          # Disease lifespan

# 1 distance point = 100 meters
neighbourhood_radius = 2        # Maximum distance allowed to travel for each person from his initial location

# 1 epoch = 8 hours
epochs = 100                    # Steps to perform during each people 1) travel and 2) spread the disease

radius = 1                      # Maximum radius for person to travel in single epoch
spread_radius = 1               # Disease spreading radius
infect_prob = 0.05              # Base probability for disease to transmit
death_prob = 0.02               # Death probability after disease

# (x, z, S, j, u)
transmitters_test_quota = 0     # Number of tests for visible transmitters to have possibility to move to quarantine zone with less death rate
others_test_quota = 1           # Number of tests for others to have possibility to move to quarantine zone with less death rate
quarantine_zone_size = 100      # Quarantine zone's capacity
remote_workers = 0.2            # Fraction of remote workers
responsible_people = 0.2        # Fraction of responsible people (which have lower probability of getting ill)

# Path to plot of disease (exposure) matrix before transmitting a disease in each epoch
plot_disease_matrix = r'C:\Users\Владислав\Downloads\Transport-networks-main\radius_{}_spread_radius_{}_infected_prob_{}'.format(str(radius).replace('.', '_'),
                                                                                                           str(spread_radius).replace('.', '_'),
                                                                                                           str(infect_prob).replace('.', '_'))

# Perform simulation (without city plots in each epoch)
timer_dict, healthy_tracker, infected_tracker, invisible_transmitters_tracker, \
transmitters_tracker, cured_tracker, dead_tracker, quarantine_tracker = simulate_transportations_with_infections(init_transmitters_num, \
                                                                                                remote_workers, \
                                                                                                responsible_people, \
                                                                                                timer_min, \
                                                                                                timer_max, \
                                                                                                transmission_time, \
                                                                                                neighbourhood_radius, \
                                                                                                infect_prob, \
                                                                                                death_prob, \
                                                                                                radius, \
                                                                                                spread_radius, \
                                                                                                quarantine_zone_size, \
                                                                                                transmitters_test_quota, \
                                                                                                others_test_quota, \
                                                                                                epochs, \
                                                                                                None)

# Perform simulation (with city plots in each epoch)
# timer_dict, healthy_tracker, infected_tracker, invisible_transmitters_tracker, \
# transmitters_tracker, cured_tracker, dead_tracker, quarantine_tracker = simulate_transportations_with_infections(init_transmitters_num, \
#                                                                                                 remote_workers, \
#                                                                                                 responsible_people, \
#                                                                                                 timer_min, \
#                                                                                                 timer_max, \
#                                                                                                 transmission_time, \
#                                                                                                 neighbourhood_radius, \
#                                                                                                 infect_prob, \
#                                                                                                 death_prob, \
#                                                                                                 radius, \
#                                                                                                 spread_radius, \
#                                                                                                 quarantine_zone_size, \
#                                                                                                 transmitters_test_quota, \
#                                                                                                 others_test_quota, \
#                                                                                                 epochs, \
#                                                                                                 plot_disease_matrix)

SAVE_DATA = True  # Flag to save data below (dataframe + plots)

# Save detailed stats
df = pd.DataFrame(np.array([healthy_tracker, infected_tracker, invisible_transmitters_tracker, \
                            transmitters_tracker, cured_tracker, dead_tracker, quarantine_tracker]).T,
                  columns=['healthy', 'infected', 'invisible_transmitters', 'transmitters', 'cured', 'dead', 'quarantined'])
if SAVE_DATA:
    df.to_csv(os.path.join(plot_disease_matrix, 'stats.csv'), sep='\t')

# Save progress plot
fig, ax = plt.subplots(nrows=2, figsize=(15, 2 * 5))

ax[0].plot(infected_tracker, '.-', c='tab:blue', label='Infected people')
ax[0].plot(invisible_transmitters_tracker + transmitters_tracker, '.-', c='tab:red', label='All transmitters (visible + invisible)')
ax[0].plot(dead_tracker, '.-', c='black', label='Deceased')
ax[0].plot(cured_tracker, '.-', c='tab:green', label='Cured')
ax[0].plot(quarantine_tracker, '.-', c='tab:orange', label='Quarantined')

ax[0].set_ylabel('People')
ax[0].set_xlabel('Time')
ax[0].set_title(f'People by groups vs. time (radius={radius}, spread_radius={spread_radius}, infect_prob={infect_prob})')
ax[0].grid()
ax[0].legend()

ax[1].plot(infected_tracker + invisible_transmitters_tracker + transmitters_tracker + dead_tracker + cured_tracker, '.-',
           c='violet', label='People that contacted a disease')

ax[1].set_ylabel('People')
ax[1].set_xlabel('Time')
ax[1].set_title(f'People that contacted a disease vs. time (radius={radius}, spread_radius={spread_radius}, infect_prob={infect_prob})')
ax[1].grid()
ax[1].legend()

if SAVE_DATA:
    fig.savefig(os.path.join(plot_disease_matrix, 'stats_ts.png'), dpi=300);

plt.show()