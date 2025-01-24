import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_mean_erp(trial_points, ecog_data):
    """
    Calculate the mean Event-Related Potential (ERP) for each finger based on trial data
    and ECoG brain signals.

    Parameters:
    trial_points (str): Path to the CSV file containing trial data with columns for
                         starting point, peak point, and finger number.
    ecog_data (str): Path to the CSV file containing ECoG brain data.

    Returns:
    numpy.ndarray: A 5x1201 matrix of the mean ERP for each of the 5 fingers.

    The function also displays a plot of the ERP for each finger.
    """

    # Load trial points and ECoG data from CSV files
    trial_points = pd.read_csv(
        trial_points, header=None, names=["starting_point", "peak_point", "finger"]
    )
    ecog_data = pd.read_csv(ecog_data, header=None, dtype=float)

    # Ensure trial points are numeric, drop rows with errors, and convert to integers
    trial_points_cor = (
        trial_points.apply(pd.to_numeric, errors="coerce").dropna().astype(int)
    )

    # Define the window length: 200 ms before and 1000 ms after the movement
    pre_movement = 200
    post_movement = 1000
    window_length = pre_movement + post_movement + 1  # Total of 1201 time points

    # Initialize an array to store the averaged ERP for each finger
    fingers_erp_mean = np.zeros((5, window_length))

    # Process each finger (1-5) and calculate its mean ERP
    for finger in range(1, 6):
        # Get the trials for the current finger
        finger_trials = trial_points_cor[trial_points_cor["finger"] == finger]
        finger_erp_data = []

        # Extract the ERP data for each trial of the current finger
        for _, trial in finger_trials.iterrows():
            start_idx = trial["starting_point"]
            window_start = start_idx - pre_movement
            window_end = start_idx + post_movement

            # Check if the window is within the bounds of the ECoG data
            if 0 <= window_start < len(ecog_data) and window_end < len(ecog_data):
                window_data = ecog_data.iloc[window_start : window_end + 1, 0].values
                finger_erp_data.append(window_data)

        # Average the ERP data across all trials for the current finger
        if finger_erp_data:
            fingers_erp_mean[finger - 1, :] = np.mean(finger_erp_data, axis=0)

    # Create a time vector from -200 ms to 1000 ms (relative to the movement start)
    time_vector = np.linspace(-pre_movement, post_movement, window_length)

    # Define pastel colors for the plot
    colors = ["lightblue", "lightpink", "lightsalmon", "lightgreen", "khaki"]

    # Plot the ERP for all fingers on a single graph
    plt.figure(figsize=(12, 6))  # Set figure size
    for i, finger in enumerate(range(1, 6)):
        plt.plot(
            time_vector,
            fingers_erp_mean[i, :],
            label=f"Finger {finger}",
            color=colors[i],
        )

    # Set axis labels and title
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Event-Related Potential (ERP) for Each Finger")

    # Add a legend and grid to the plot
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("finger_erp_mean_graph.png")
    # Show the plot
    plt.show()

    # Format and print the ERP matrix
    fingers_erp_df = pd.DataFrame(
        fingers_erp_mean,
        index=[f"Finger {i+1}" for i in range(5)],
        columns=[f"Time {i}" for i in range(window_length)],
    )
    print(fingers_erp_df)

    # Return the averaged ERP data
    return fingers_erp_mean


# Function usage
trial_points = r"C:\Users\Home\Desktop\Studies\Phyton\projects 2024-2025\Miniproject_2\data\events_file_ordered.csv"
ecog_data_file = r"C:\Users\Home\Desktop\Studies\Phyton\projects 2024-2025\Miniproject_2\data\brain_data_channel_one.csv"
fingers_erp_mean = calc_mean_erp(trial_points, ecog_data_file)
