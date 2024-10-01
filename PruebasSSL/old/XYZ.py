# colour plot
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Parameters
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # number of channels or microphones
RATE = 48000  # sampling rate
CHUNK = int(0.2 * RATE)  # buffer size for 100 ms
RECORD_SECONDS = 900  # recording time
d = 0.04  # distance between microphones
c = 343  # speed of sound in m/s
device_index = 1

# Microphone positions in 3D
mic_positions = np.array([
    [0.12 * np.cos(np.radians(0)), 0.12 * np.sin(np.radians(0)), 0.5],  # Mic 1
    [0.12 * np.cos(np.radians(120)), 0.12 * np.sin(np.radians(120)), 0.5],  # Mic 2
    [0.12 * np.cos(np.radians(240)), 0.12 * np.sin(np.radians(240)), 0.5],  # Mic 3
    [0.2 * np.cos(np.radians(0)), 0.2 * np.sin(np.radians(0)), 0.25],  # Mic 4
    [0.2 * np.cos(np.radians(120)), 0.2 * np.sin(np.radians(120)), 0.25],  # Mic 5
    [0.2 * np.cos(np.radians(240)), 0.2 * np.sin(np.radians(240)), 0.25]  # Mic 6
])

'''
# Microphone positions in 3D
mic_positions = np.array([
    [d, 0, 0],  # Mic 1
    [2*d, 0, 0],  # Mic 2
    [0, d, 0],  # Mic 3
    [0, 2*d, 0],  # Mic 4
    [0, 0, d],  # Mic 5
    [0, 0, 2*d]  # Mic 6
])
'''

# Beamforming using FFT
def beamform_fft(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    freq_domain_signal = fft(signal_data, axis=0)

    # Frequency vector
    freqs = np.fft.fftfreq(num_samples, 1 / RATE)
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            delays = np.dot(mic_positions, direction_vector) / c
            delay_phase_shifts = np.exp(-2j * np.pi * freqs[:, np.newaxis] * delays)

            # Apply phase shifts in the frequency domain
            beamformed_signal_freq = np.sum(freq_domain_signal * delay_phase_shifts, axis=1)
            beamformed_signal_time = ifft(beamformed_signal_freq, axis=0)

            # Calculate energy
            energy[az_idx, el_idx] = np.sum(np.abs(beamformed_signal_time)**2)

    return energy

# Audio setup
audio = pyaudio.PyAudio()

# Create the stream object for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []
azimuth_range = np.arange(-90, 91, 20)  # Azimuth angles from -90° to 90°
elevation_range = np.arange(-90, 91, 20)  # Elevation angles from -90° to 90°

# Set up real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Beamformed Energy')

for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binary data to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

    # Calculate energy using FFT-based beamforming
    energy = beamform_fft(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Estimated angle: Azimuth = {estimated_azimuth:.2f}°, Elevation = {estimated_elevation:.2f}°")

    # Update the heatmap data
    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Update color limits
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the complete recording to a .wav file
with wave.open("output_pyaudio_realtime3D_6ch.wav", 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Recording saved in output_pyaudio_realtime3D_6ch.wav")
