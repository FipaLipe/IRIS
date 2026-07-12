import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from .image_utils import image_to_pulses
from .constants import (
    THRESHOLD_VOLTAGE,
    SCALE_VOLTAGE,
    CURVATURE,
    TIME_RESOLUTION,
    IMAGE_THRESHOLD,
    NEURON_COUNT,
    BETA,
    TAU,
    K,
    FINAL_COLS,
    VOLTAGE,
    MAX_ITERATION,
    LEARNING_RATE,
)


def IGT_exponent(
    v,
    threshold_voltage=THRESHOLD_VOLTAGE,
    scale_voltage=SCALE_VOLTAGE,
    curvature=CURVATURE,
):
    return np.exp(
        -(
            (np.maximum(0, (np.absolute(v) - threshold_voltage)) / scale_voltage)
            ** curvature
        )
    )


def evolve_IGT(k, v, In_1, beta, tau, time_resolution):
    one_vector = np.ones_like(In_1)
    current_in_infinity = k * (one_vector - IGT_exponent(v))

    return current_in_infinity + (In_1 - current_in_infinity) * np.exp(
        -((time_resolution / tau) ** beta)
    )


def current_to_voltage(
    current_in_infinity,
    k,
    threshold_voltage=THRESHOLD_VOLTAGE,
    scale_voltage=SCALE_VOLTAGE,
    curvature=CURVATURE,
):
    return (
        np.power(
            np.log(1 / np.maximum(1e-9, (1 - (current_in_infinity / k)))), 1 / curvature
        )
        * scale_voltage
        + threshold_voltage
    )


def run_reservoir(
    images,
    labels,
    normalized_reservoir,
    reservoir_state,
    weight_input,
    weight_output,
    bias_output=np.array([]),
    time_resolution=TIME_RESOLUTION,
    mode="TRAIN",
    image_threshold=IMAGE_THRESHOLD,
    neuron_count=NEURON_COUNT,
    scaler=None,
    pca=None,
    max_iterations=MAX_ITERATION,
):

    # G = nx.from_numpy_array(np.abs(weight_reservoir), create_using=nx.DiGraph)
    # pos = nx.spring_layout(G, seed=42)

    # fig, ax = plt.subplots(figsize=(6, 6))

    states = []
    predicted_results = []

    confusion_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    correct_predictions_count = 0

    current_state = reservoir_state.copy()

    for i, k in enumerate(images):
        snapshots_diff = []
        prev = None
        for n, data in enumerate(k):
            data = data.reshape(14, 1)

            result_current = normalized_reservoir @ current_state

            current_state = evolve_IGT(
                K,
                current_to_voltage(result_current, K) + (weight_input @ data),
                current_state,
                BETA,
                TAU,
                time_resolution,
            )

            if n in (3, 7, 10):
                snap = current_state.flatten().copy()
                snapshots_diff.append(snap - prev if prev is not None else snap)
                prev = snap

        states.append(np.concatenate([current_state.flatten()] + snapshots_diff))
        predicted_results.append(labels[i])

        if mode == "TEST":
            state_scaled = scaler.transform(states[i].reshape(1, -1))
            state_reduced = pca.transform(state_scaled)

            y = (weight_output @ state_reduced.reshape(-1, 1)) + bias_output.reshape(
                -1, 1
            )

            predicted_label = np.argmax(y)
            expected_label = np.argmax(labels[i])

            confusion_matrix[expected_label][predicted_label] += 1

            correct_predictions_count += expected_label == predicted_label

        # for _ in range(FLUSH_TIME):
        #     result_current = normalized_reservoir @ current_state

        #     current_state = evolve_IGT_v(
        #         K,
        #         current_to_voltage_v(result_current, K) + (weight_input @ zeros_flush),
        #         current_state,
        #         BETA,
        #         TAU
        #     )

        current_state = np.zeros((neuron_count, 1))

    predicted_results = np.array(predicted_results)

    if mode == "TRAIN":
        scaler = StandardScaler()
        pca = PCA(n_components=min(0.95, 100), whiten=True, random_state=42)

        states_scaled = scaler.fit_transform(states)
        states_reduced = pca.fit_transform(states_scaled)

        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 6), max_iter=max_iterations, n_jobs=-1
        )  # , multi_class='multinomial'
        clf.fit(states_reduced, labels.argmax(axis=1))

        weight_output = clf.coef_
        bias_output = clf.intercept_

        return weight_output, bias_output, current_state, scaler, pca

    elif mode == "TEST":
        confusion_matrix = confusion_matrix / confusion_matrix.sum(
            axis=1, keepdims=True
        )

        return correct_predictions_count, confusion_matrix
