import wave
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import time
from mpl_toolkits.mplot3d import Axes3D

# Path to the uploaded WAV file

file_path = '/Users/30068385/OneDrive - Western Sydney University/SSL/merged_F6_white_noise_1m.wav'

#file_path = '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_1_sync.wav'

# Beamforming parameters
speed_of_sound = 343  # Speed of sound in air (m/s)
array_diameter = 0.14  # Diameter of the circular array (meters)
radius = array_diameter / 2
azimuth_step = 1  # Azimuth step in degrees
window_duration = 0.2  # Window size in seconds (200 ms)
overlap = 0.5  # 50% overlap
azimuth_angles = np.arange(0, 360, azimuth_step)

# Open the WAV file
wav_file = wave.open(file_path, 'rb')

# Extract parameters
n_channels = wav_file.getnchannels()
sampwidth = wav_file.getsampwidth()
framerate = wav_file.getframerate()
n_frames = wav_file.getnframes()

# Read all frames
frames = wav_file.readframes(n_frames)
wav_file.close()

# Convert frames to numpy array
wave_data = np.frombuffer(frames, dtype=np.int16)

# Reshape wave_data according to the number of channels
wave_data = wave_data.reshape(-1, n_channels)

# Analysis parameters
window_size_samples = int(window_duration * framerate)
step_size = int(window_size_samples * (1 - overlap))
n_windows = (n_frames - window_size_samples) // step_size + 1

# Create a circular array of microphones
mic_positions = pra.circular_2D_array([0, 0], n_channels, 0, radius)

mic_positions = np.c_[
    [0.07, 0, 1],  # mic 1
    [0.035, 0.06062, 1],  # mic 2
    [-0.035, 0.06062, 1],  # mic 2
    [-0.07, 0, 1],  # mic 2
    [-0.035, -0.06062, 1],  # mic 2
    [0.035, -0.06062, 1]  # mic 2
]

a = [0, 120, 240]

h = [0.4 , 0.01]
r = [0.45, 0.65]

mic_positions2 = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]]  # Mic 6
]).T

'''
a = [0, -120, -240]
# config 1 equidistance
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

# config 2 augmented
#h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
#r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]


mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],  # Mic 6
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],  # Mic 7
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # Mic 8
    [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],  # Mic 9
    [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],  # Mic 10
    [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],  # Mic 11
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],  # Mic 12
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],  # Mic 13
    [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],  # Mic 14
    [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],  # Mic 15
    [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],  # Mic 16
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],  # Mic 17
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]] # Mic 18
])
'''


fs = framerate
nfft = 1024  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# Define the SSL algorithms to use
selected_algorithms = {
    'SRP': pra.doa.SRP(mic_positions, fs, nfft, c=speed_of_sound, colatitude=None, dim=3),
    'NormMUSIC': pra.doa.NormMUSIC(mic_positions, fs, nfft, c=speed_of_sound, colatitude=None, dim=3)
}

# Initialize storage for power estimates and execution times
power_estimates = {algo_name: [] for algo_name in selected_algorithms.keys()}
execution_times = {algo_name: [] for algo_name in selected_algorithms.keys()}
beamformed_energy = []
time_axis = []

# Iterate over the signal with sliding windows
for window_idx in range(n_windows):
    start_idx = window_idx * step_size
    end_idx = start_idx + window_size_samples
    window_signal = wave_data[start_idx:end_idx, :].T  # Transpose to shape (num_mics, samples)

    # Compute STFT for each channel
    X = np.array([pra.transform.stft.analysis(signal, nfft, nfft // 2).T for signal in window_signal])

    # Apply each SSL algorithm
    for algo_name, doa in selected_algorithms.items():
        start_time = time.time()  # Start timing
        doa.locate_sources(X, freq_bins=freq_bins)
        end_time = time.time()  # End timing

        # Calculate execution time
        exec_time = end_time - start_time
        execution_times[algo_name].append(exec_time)

        # Store the estimated power spectrum over all directions
        power_estimates[algo_name].append(doa.grid.values)

    # Store the time axis
    time_axis.append(start_idx / fs)

# Convert power estimates to numpy arrays for easier manipulation
power_estimates = {algo_name: np.array(power) for algo_name, power in power_estimates.items()}

# Visualization
fig, axes = plt.subplots(len(selected_algorithms), 1, figsize=(12, 8))

for idx, (algo_name, power) in enumerate(power_estimates.items()):
    # Plot the power estimates as a heatmap for each algorithm
    axes[idx].imshow(np.array(power).T, aspect='auto', extent=[time_axis[0], time_axis[-1], azimuth_angles[-1], azimuth_angles[0]], cmap='inferno')
    axes[idx].set_title(f'{algo_name} Energy Level per Azimuth Angle over Time')
    axes[idx].set_xlabel('Time [s]')
    axes[idx].set_ylabel('Azimuth Angle [degrees]')
    axes[idx].invert_yaxis()

plt.tight_layout()


# Execution time plot
plt.figure(figsize=(10, 5))
for algo_name, exec_times in execution_times.items():
    plt.plot(time_axis, exec_times, label=algo_name)
plt.title('Execution Time per Window for Different Algorithms')
plt.xlabel('Time [s]')
plt.ylabel('Execution Time [s]')
plt.legend()
plt.grid(True)


# 3D plot for SRP power estimates (example)
X_3D, Y_3D = np.meshgrid(time_axis, azimuth_angles)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Example: plot 3D surface for SRP
Z_3D = np.array(power_estimates['SRP']).T  # Take SRP power estimates

# Plot the surface
surf = ax.plot_surface(X_3D, Y_3D, Z_3D, cmap='inferno', edgecolor='none', alpha=0.8)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Power')

# Set labels and title
ax.set_title('3D Power Estimates (SRP) over Time and Azimuth Angles')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Azimuth Angle [degrees]')
ax.set_zlabel('Power')

# Set viewing angle for better visualization
ax.view_init(elev=30, azim=120)

plt.tight_layout()

# 3D plot for NormMUSIC power estimates
X_3D, Y_3D = np.meshgrid(time_axis, azimuth_angles)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D surface for NormMUSIC
Z_3D = np.array(power_estimates['NormMUSIC']).T  # Take NormMUSIC power estimates

# Plot the surface
surf = ax.plot_surface(X_3D, Y_3D, Z_3D, cmap='inferno', edgecolor='none', alpha=0.8)

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Power')

# Set labels and title
ax.set_title('3D Power Estimates (NormMUSIC) over Time and Azimuth Angles')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Azimuth Angle [degrees]')
ax.set_zlabel('Power')

# Set viewing angle for better visualization
ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()

