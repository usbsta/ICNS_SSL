import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import .npz data
energy_path = '/Users/bjrn/OneDrive - Western Sydney University/Archive/energy_data.npz'
prob_path = '/Users/bjrn/OneDrive - Western Sydney University/Archive/probabilities_data.npz'

# Load the .npz files
data_energy = np.load(energy_path)
data_prob = np.load(prob_path)
energy_keys = list(data_energy.keys())
prob_keys = list(data_prob.keys())

# Extract the energy and probability data arrays
energy_data_array = data_energy[energy_keys[0]]  # Shape: (frames, columns, rows)
prob_data_array = data_prob[prob_keys[0]]        # Shape: (frames, columns, rows)

# Check the shapes
print("Energy data shape:", energy_data_array.shape)
print("Probability data shape:", prob_data_array.shape)

# Get the number of frames, columns, and rows
num_frames, num_columns, num_rows = energy_data_array.shape

# Create matplotlib figures
plt.ion()

# Set up the extent if you want to map the data to specific coordinates
extent = [-180, 180, -90, 90]  # Adjust these values as needed

# Figure for energy heatmap
fig_energy, ax_energy = plt.subplots()
fig_energy.canvas.manager.set_window_title('Energy Heatmap')
energy_display = ax_energy.imshow(np.zeros((num_columns, num_rows)), cmap=cm.jet, aspect='auto', origin='lower', extent=extent)
ax_energy.set_title('Energy SSL')

# Figure for probability heatmap
fig_prob, ax_prob = plt.subplots()
fig_prob.canvas.manager.set_window_title('Probability Heatmap')
prob_display = ax_prob.imshow(np.zeros((num_columns, num_rows)), cmap=cm.jet, aspect='auto', origin='lower', extent=extent)
ax_prob.set_title('CNN SSL')

# Loop over each frame to update the heatmaps
for data_index in range(num_frames):
    # Extract the data for the current frame
    synchronized_energy = energy_data_array[data_index, :, :].T  # Shape: (columns, rows)
    synchronized_prob = prob_data_array[data_index, :, :].T      # Shape: (columns, rows)

    # Optionally transpose if the image appears rotated
    # synchronized_energy = synchronized_energy.T
    # synchronized_prob = synchronized_prob.T

    # Update the heatmaps with the current data
    energy_display.set_data(synchronized_energy)
    prob_display.set_data(synchronized_prob)

    # Adjust color limits if needed
    energy_display.set_clim(vmin=np.min(synchronized_energy), vmax=np.max(synchronized_energy))
    prob_display.set_clim(vmin=np.min(synchronized_prob), vmax=np.max(synchronized_prob))

    # Redraw the figures
    fig_energy.canvas.draw()
    fig_prob.canvas.draw()

    plt.pause(0.1)  # Pause to update the plots; adjust as needed

# Keep the plots open after the loop finishes
plt.ioff()
plt.show()
