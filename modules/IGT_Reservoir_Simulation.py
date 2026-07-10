import numpy as np
from sklearn.linear_model import RidgeCV
from matplotlib import pyplot as plt
from .image_utils import image_to_pulses
from .constants import THRESHOLD_VOLTAGE, SCALE_VOLTAGE, CURVATURE, TIME_RESOLUTION, IMAGE_THRESHOLD, NEURON_COUNT, BETA, TAU, K, FINAL_COLS, VOLTAGE

def IGT_exponent(v, threshold_voltage=THRESHOLD_VOLTAGE, scale_voltage=SCALE_VOLTAGE, curvature=CURVATURE):
    return np.exp(-(np.maximum(0,(np.absolute(v) - threshold_voltage)) / scale_voltage) ** curvature)

def evolve_IGT(k, v, In_1, beta, tau, time_resolution):
    one_vector = np.ones_like(In_1)
    current_in_infinity = (k * (one_vector - IGT_exponent(v)))

    return current_in_infinity + (In_1 - current_in_infinity) * np.exp(-(time_resolution / tau) ** beta)

def current_to_voltage(current_in_infinity, k, threshold_voltage=THRESHOLD_VOLTAGE, scale_voltage=SCALE_VOLTAGE, curvature=CURVATURE):
    return np.power(np.log(1/np.maximum(1e-9,(1-(current_in_infinity/k)))), 1/curvature) * scale_voltage + threshold_voltage

def run_reservoir(
        images,
        labels,
        normalized_reservoir,
        reservoir_state,
        weight_input,
        weight_output,
        zeros_flush,
        bias_output = np.array([]),
        mode = "TRAIN",
        image_threshold=IMAGE_THRESHOLD,
        neuron_count=NEURON_COUNT,
        k=K,
        beta=BETA,
        tau=TAU,
        time_resolution=TIME_RESOLUTION,
        final_cols=FINAL_COLS,
        voltage=VOLTAGE,
        binarize=True,
        dtype=np.uint8
    ):

    # G = nx.from_numpy_array(np.abs(weight_reservoir), create_using=nx.DiGraph)
    # pos = nx.spring_layout(G, seed=42)

    # fig, ax = plt.subplots(figsize=(6, 6))

    states = []
    predicted_results = []

    confusion_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    correct_predictions_count = 0

    current_state = reservoir_state.copy()

    for i, img in enumerate(images):
        image = image_to_pulses(img, intervals=1, threshold=image_threshold, binarize=binarize, dtype=dtype) * voltage

        for data in image:
            data = data.reshape(data.size, 1)

            result_current = normalized_reservoir @ current_state

            current_state = evolve_IGT(
                k, 
                current_to_voltage(result_current, k) + (weight_input @ data), 
                current_state, 
                beta, 
                tau,
                time_resolution
            )

            # ax.clear()
            # node_colors = current_state.flatten()
            # nx.draw(
            #     G, pos, ax=ax,
            #     node_color=node_colors, cmap=plt.cm.viridis,
            #     with_labels=False, node_size=200, arrows=False
            # )
            # clear_output(wait=True)
            # display(fig)
        
        for _ in range(final_cols):
            result_current = normalized_reservoir @ current_state

            current_state = evolve_IGT(
                k, 
                current_to_voltage(result_current, k) + (weight_input @ zeros_flush), 
                current_state, 
                beta, 
                tau,
                time_resolution
            )

        # print(f"add {current_state.flatten()} nos {states}")
        states.append(current_state.flatten())
        predicted_results.append(labels[i])
        
        if mode == "TEST":
            y = (weight_output @ current_state) + bias_output.reshape(-1, 1)

            predicted_label = np.argmax(y)
            expected_label = np.argmax(labels[i])

            confusion_matrix[expected_label][predicted_label] += 1

            correct_predictions_count += (expected_label == predicted_label)

        # for _ in range(FLUSH_TIME):
        #     result_current = normalized_reservoir @ current_state

        #     current_state = evolve_IGT(
        #         K, 
        #         current_to_voltage(result_current, K) + (weight_input @ zeros_flush), 
        #         current_state, 
        #         beta, 
        #         tau,
        #         time_resolution
        #     )
        
        # print(np.average(current_state))

        current_state = np.zeros((neuron_count, 1))

    states = np.array(states)
    predicted_results = np.array(predicted_results)

    if mode == "TRAIN":
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(states, predicted_results)

        weight_output = ridge.coef_
        bias_output = ridge.intercept_

        return weight_output, bias_output, current_state
    
    if mode == "TEST":
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        return correct_predictions_count, confusion_matrix