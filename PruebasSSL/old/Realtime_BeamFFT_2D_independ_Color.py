import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Parameters
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # number of channels or microphones
RATE = 48000  # sampling rate
CHUNK = int(0.2 * RATE)  # buffer size in 200 ms
RECORD_SECONDS = 900  # recording time
#OUTPUT_FILENAME = "output_pyaudio_realtime6_fft_heatmap.wav"
d = 0.04  # distance between microphones
c = 343  # speed of sound in m/s

# Microphone positions in 1D for azimuth and elevation
mic_positions_azimuth = np.array([
    [0, 0, 0],  # Mic 1
    [d, 0, 0],  # Mic 2
    [2*d, 0, 0]  # Mic 3
])

# diferent order because angle of elevation is inverse of azimuth
mic_positions_elevation = np.array([
    [0, 0, 2*d],  # Mic 4
    [0, 0, d],   # Mic 5
    [0, 0, 0]  # Mic 6
])

# Audio setup
audio = pyaudio.PyAudio()
device_index = 0

# Create the stream object for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []
energy_azimuth_history = []
energy_elevation_history = []

azimuth_range = np.arange(-90, 91, 5)  # azimuth angles in degrees
elevation_range = np.arange(-90, 91, 5)  # elevation angles in degrees

# Setup for real-time plotting
plt.ion()
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot for energy vs azimuth
line_energy_azimuth, = ax1.plot(azimuth_range, np.zeros(len(azimuth_range)))
ax1.set_ylim(0, 1)
ax1.set_xlabel('Azimuth Angle (degrees)')
ax1.set_ylabel('Energy')

# Plot for energy vs elevation
line_energy_elevation, = ax2.plot(elevation_range, np.zeros(len(elevation_range)))
ax2.set_ylim(0, 1)
ax2.set_xlabel('Elevation Angle (degrees)')
ax2.set_ylabel('Energy')

# Setup for the separate heatmap figure
fig2, ax3 = plt.subplots(figsize=(10, 8))
heatmap = ax3.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                     extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                     origin='lower', aspect='auto', cmap='viridis')
fig2.colorbar(heatmap, ax=ax3, label='Energy')
ax3.set_xlabel('Azimuth Angle (degrees)')
ax3.set_ylabel('Elevation Angle (degrees)')
ax3.set_title('Real-Time Energy Heatmap')

# Capture and real-time processing
for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binary data to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]  # number of samples
    freq_domain_signal = fft(signal_data, axis=0)  # FFT of all channels

    freqs = np.fft.fftfreq(num_samples, 1 / RATE)  # Frequency vector

    energy_azimuth = np.zeros(len(azimuth_range))  # energy at each azimuth angle
    energy_elevation = np.zeros(len(elevation_range))  # energy at each elevation angle

    # Iterate over all azimuth angles (using the first three microphones)
    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        # Calculate the phase shifts for azimuth
        delays = []
        for pos in mic_positions_azimuth:
            x, y, z = pos
            delay = (x * np.sin(azimuth_rad)) / c
            delays.append(delay)
        delay_phases = np.exp(-2j * np.pi * freqs[:, np.newaxis] * np.array(delays))

        # Apply Delay and Sum Beamforming in frequency domain
        beamformed_signal_freq = np.sum(freq_domain_signal[:, :3] * delay_phases, axis=1)
        output_signal = ifft(beamformed_signal_freq)

        # Calculate the energy of the combined signal
        energy_azimuth[az_idx] = np.sum(np.abs(output_signal)**2)

    # Iterate over all elevation angles (using the last three microphones)
    for el_idx, phi in enumerate(elevation_range):
        elevation_rad = np.radians(phi)

        # Calculate the phase shifts for elevation
        delays = []
        for pos in mic_positions_elevation:
            x, y, z = pos
            delay = (z * np.sin(elevation_rad)) / c
            delays.append(delay)
        delay_phases = np.exp(-2j * np.pi * freqs[:, np.newaxis] * np.array(delays))

        # Apply Delay and Sum Beamforming in frequency domain
        beamformed_signal_freq = np.sum(freq_domain_signal[:, 3:6] * delay_phases, axis=1)
        output_signal = ifft(beamformed_signal_freq)

        # Calculate the energy of the combined signal
        energy_elevation[el_idx] = np.sum(np.abs(output_signal)**2)

    # Save the energy for each angle and time
    energy_azimuth_history.append(energy_azimuth)
    energy_elevation_history.append(energy_elevation)

    # Create a 2D array for the heatmap
    energy_heatmap = np.outer(energy_elevation, energy_azimuth)

    # Update real-time energy vs azimuth plot
    line_energy_azimuth.set_ydata(energy_azimuth)
    ax1.set_ylim(0, np.max(energy_azimuth) * 1.1)

    # Update real-time energy vs elevation plot
    line_energy_elevation.set_ydata(energy_elevation)
    ax2.set_ylim(0, np.max(energy_elevation) * 1.1)

    # Update the heatmap in the separate figure
    heatmap.set_data(energy_heatmap)
    heatmap.set_clim(vmin=np.min(energy_heatmap), vmax=np.max(energy_heatmap))  # Update color limits

    # Find the azimuth and elevation angles with the maximum energy
    max_energy_idx = np.unravel_index(np.argmax(energy_heatmap), energy_heatmap.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[1]]
    estimated_elevation = elevation_range[max_energy_idx[0]]
    print(f"Estimated Angle: Azimuth = {estimated_azimuth:.2f}°, Elevation = {estimated_elevation:.2f}°")

    # Draw and update all plots
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    plt.pause(0.01)

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the complete recording to a .wav file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Recording saved in {OUTPUT_FILENAME}")

plt.ioff()
plt.show()
