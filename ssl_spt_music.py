import wave
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import time
from mpl_toolkits.mplot3d import Axes3D

# Path to the uploaded WAV file

file_path = '/Users/30068385/OneDrive - Western Sydney University/SSL/merged_F6_white_noise_1m.wav'

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


fs = framerate
nfft = 1024  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# Define the SSL algorithms to use
selected_algorithms = {
    'SRP': pra.doa.SRP(mic_positions, fs, nfft, c=speed_of_sound),
    'NormMUSIC': pra.doa.NormMUSIC(mic_positions, fs, nfft, c=speed_of_sound)
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

