import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set up the 360 video input
video_path = '/Users/bjrn/OneDrive - Western Sydney University/360 videos/3rd Oct 11/5.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Verify if the video was opened successfully
ret, frame = cap.read()
if not ret:
    print("Error reading frame from video.")
    exit()

height, width, _ = frame.shape

# Import .npz data
energy_path = '/Users/bjrn/OneDrive - Western Sydney University/Archive/energy_data.npz'
prob_path = '/Users/bjrn/OneDrive - Western Sydney University/Archive/probabilities_data.npz'

data_energy = np.load(energy_path)
data_prob = np.load(prob_path)
energy_keys = list(data_energy.keys())
prob_keys = list(data_prob.keys())

data_fps = 10
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)

# Create matplotlib figures
plt.ion()

# Figure for video
fig_video, ax_video = plt.subplots()
fig_video.canvas.manager.set_window_title('Video Frame')
frame_display = ax_video.imshow(np.zeros((height, width, 3), dtype=np.uint8))

# Set grid and labels for video
ax_video.set_xticks(np.linspace(0, width, 5))
ax_video.set_xticklabels(['-180', '-90', '0', '90', '180'], color='black')
ax_video.set_yticks(np.linspace(0, height, 5))
ax_video.set_yticklabels(['90', '45', '0', '-45', '-90'], color='black')
ax_video.grid(color='black', linestyle='--', linewidth=0.5)

# Set up the extent based on your coordinate system
extent = [-180, 180, -90, 90]  # Adjust these values as needed

# Figure for energy heatmap
fig_energy, ax_energy = plt.subplots()
fig_energy.canvas.manager.set_window_title('Energy Heatmap')
energy_display = ax_energy.imshow(np.zeros((46, 181)), cmap=cm.jet, extent=extent, aspect='auto', origin='lower')
ax_energy.set_title('Energy SSL')

# Figure for probability heatmap
fig_prob, ax_prob = plt.subplots()
fig_prob.canvas.manager.set_window_title('Probability Heatmap')
prob_display = ax_prob.imshow(np.zeros((46, 181)), cmap=cm.jet, extent=extent, aspect='auto', origin='lower')
ax_prob.set_title('CNN SSL')

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Calculate the corresponding index for the .npz data
    if frame_idx % int(video_fps / data_fps) == 0:
        data_index = frame_idx // int(video_fps / data_fps)

        # Ensure data_index is within bounds for both energy and probability data
        if data_index < len(data_energy[energy_keys[0]]) and data_index < len(data_prob[prob_keys[0]]):
            # Reshape the data correctly
            synchronized_energy = data_energy[energy_keys[0]][data_index, :].reshape((46, 181))
            synchronized_prob = data_prob[prob_keys[0]][data_index, :].reshape((46, 181))

            # Optionally transpose if the image appears rotated
            # synchronized_energy = synchronized_energy.T
            # synchronized_prob = synchronized_prob.T

            # Update the heatmaps with the synchronized data
            energy_display.set_data(synchronized_energy)
            prob_display.set_data(synchronized_prob)

            # Set dynamic limits for the heatmaps
            energy_display.set_clim(vmin=np.min(synchronized_energy), vmax=np.max(synchronized_energy))
            prob_display.set_clim(vmin=np.min(synchronized_prob), vmax=np.max(synchronized_prob))

            # Redraw the heatmaps
            fig_energy.canvas.draw()
            fig_prob.canvas.draw()

    # Roll the frame by half of the width
    roll_width = width // 2
    frame = np.roll(frame, shift=roll_width, axis=1)

    # Update the frame in the plot
    frame_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fig_video.canvas.draw()
    plt.pause(0.001)  # Small pause to allow the frame to update

    # Check if 'q' key is pressed to exit
    if plt.waitforbuttonpress(0.001):
        break

    frame_idx += 1
