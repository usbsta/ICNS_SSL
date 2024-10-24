import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2  # Import OpenCV

# Paths to your .npz files
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

# Create a figure with two subplots side by side
plt.ion()

fig, (ax_energy, ax_prob) = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.manager.set_window_title('Energy and Probability Heatmaps')

# Set up the extent if you want to map the data to specific coordinates
extent = [-180, 180, -90, 90]  # Adjust these values as needed

# Initialize the energy heatmap
energy_display = ax_energy.imshow(np.zeros((num_columns, num_rows)), cmap=cm.jet,
                                  aspect='auto', origin='lower', extent=extent)
ax_energy.set_title('Energy SSL')
ax_energy.set_xlabel('Azimuth (degrees)')
ax_energy.set_ylabel('Elevation (degrees)')
plt.colorbar(energy_display, ax=ax_energy, label='Energy')

# Initialize the probability heatmap
prob_display = ax_prob.imshow(np.zeros((num_columns, num_rows)), cmap=cm.jet,
                              aspect='auto', origin='lower', extent=extent)
ax_prob.set_title('CNN SSL')
ax_prob.set_xlabel('Azimuth (degrees)')
ax_prob.set_ylabel('Elevation (degrees)')
plt.colorbar(prob_display, ax=ax_prob, label='Probability')

# Get figure dimensions in pixels
dpi = fig.get_dpi()
width_inch, height_inch = fig.get_size_inches()
width_px, height_px = int(width_inch * dpi), int(height_inch * dpi)
size = (width_px, height_px)

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10  # Frames per second for the video
out = cv2.VideoWriter('energy_probability_heatmaps.mp4', fourcc, fps, size)

# Parameters for smoothing
window_size = 20  # Adjust the window size as needed
energy_frames = []  # List to store energy frames for smoothing
prob_frames = []    # List to store probability frames for smoothing

def smooth_frames(frames_list, window_size):
    if len(frames_list) < window_size:
        # Not enough frames to smooth; average over available frames
        return np.mean(frames_list, axis=0)
    else:
        # Use the last 'window_size' frames
        return np.mean(frames_list[-window_size:], axis=0)

for data_index in range(num_frames):
    # Extract the data for the current frame
    energy_frame = energy_data_array[data_index, :, :].T  # Shape: (columns, rows)
    prob_frame = prob_data_array[data_index, :, :].T      # Shape: (columns, rows)

    # Append frames to lists for smoothing
    energy_frames.append(energy_frame)
    prob_frames.append(prob_frame)

    # Apply smoothing
    smoothed_energy = smooth_frames(energy_frames, window_size)
    smoothed_prob = smooth_frames(prob_frames, window_size)

    # Optionally transpose if the image appears rotated
    # smoothed_energy = smoothed_energy.T
    # smoothed_prob = smoothed_prob.T

    # Update the heatmaps with the smoothed data
    energy_display.set_data(smoothed_energy)
    prob_display.set_data(smoothed_prob)

    # Adjust color limits to enhance visualization
    energy_display.set_clim(vmin=np.min(smoothed_energy), vmax=np.max(smoothed_energy))
    prob_display.set_clim(vmin=np.min(smoothed_prob), vmax=np.max(smoothed_prob))

    # Redraw the figures
    fig.canvas.draw()

    # Convert the canvas to an image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(height_px, width_px, 3)

    # Write the image to the video
    out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Optional: update the plot in real-time
    # plt.pause(0.001)

# Release the video writer
out.release()
print("Video saved as 'energy_probability_heatmaps.mp4'.")

# Keep the plots open after the loop finishes
plt.ioff()
plt.show()
