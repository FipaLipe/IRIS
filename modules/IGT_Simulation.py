import numpy as np
import matplotlib.pyplot as plt
from .constants import THRESHOLD_VOLTAGE, SCALE_VOLTAGE, CURVATURE, TIME_RESOLUTION

def create_pulse_train(basic_voltage, pulse_voltage, pulse_duration, interval_duration, pulse_quantity):
    train_length = (pulse_duration + interval_duration) * pulse_quantity

    pulse_train = np.zeros(train_length) + basic_voltage

    for i in range(pulse_quantity):
        pulse_train[i * (pulse_duration + interval_duration): (i * (pulse_duration + interval_duration)) + (pulse_duration)] = pulse_voltage
    
    return pulse_train

def evolve_single_IGT(k, v, In_1, beta, tau, threshold_voltage=THRESHOLD_VOLTAGE, scale_voltage=SCALE_VOLTAGE, curvature=CURVATURE, time_resolution=TIME_RESOLUTION):
    current_in_infinity = (k * (1 - np.exp(-((v - threshold_voltage) / scale_voltage) ** curvature)))

    return current_in_infinity + (In_1 - current_in_infinity) * np.exp(-(time_resolution / tau) ** beta)

def plot_out_current(pulse_train, out_current, title, ranges, time_resolution=TIME_RESOLUTION):
    time = np.arange(len(out_current)) * time_resolution

    _, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]}
    )

    # Pulse train
    ax1.step(time, pulse_train, where="post")
    ax1.set_ylabel("Pulse")
    ax1.set_title(title)
    ax1.grid(True)

    # Out Current
    ax2.plot(time, out_current)
    ax2.scatter(time, out_current, facecolors="none", edgecolors="tab:blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$I_{DS}\ (nA)$")
    ax2.grid(True)
    if ranges != ((0,0),(0,0)):
        ax2.set_xlim(ranges[0])
        ax2.set_ylim(ranges[1])

    plt.tight_layout()
    plt.show()

def generate_single_out_current(pulse_train, k, beta, tau, time_resolution, plot=False, title="", ranges=((0,0),(0,0))):
    
    out_current = np.zeros_like(pulse_train)

    for i in range(1, len(out_current)):
        out_current[i] = evolve_single_IGT(k, pulse_train[i], out_current[i-1], beta, tau, time_resolution=time_resolution)

    if plot:
        plot_out_current(pulse_train, out_current, title, ranges, time_resolution=time_resolution)


    return out_current

def current_to_voltage(current_in_infinity, k, threshold_voltage=THRESHOLD_VOLTAGE, scale_voltage=SCALE_VOLTAGE, curvature=CURVATURE):
    # if k > current_in_infinity.any():
    #     print(current_in_infinity)
        
    return np.power(np.log(1/(1-(current_in_infinity/k))), 1/curvature) * scale_voltage + threshold_voltage