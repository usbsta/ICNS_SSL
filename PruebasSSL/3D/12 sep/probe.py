import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt32
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 1200000
c = 343
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.6e9

device_index = 1

mic_positions = np.array([
    [0.12 * np.cos(np.radians(0)), 0.12 * np.sin(np.radians(0)), 0.5],  # Mic 1
    [0.12 * np.cos(np.radians(120)), 0.12 * np.sin(np.radians(120)), 0.5],  # Mic 2
    [0.12 * np.cos(np.radians(240)), 0.12 * np.sin(np.radians(240)), 0.5],  # Mic 3
    [0.2 * np.cos(np.radians(0)), 0.2 * np.sin(np.radians(0)), 0.25],  # Mic 4
    [0.2 * np.cos(np.radians(120)), 0.2 * np.sin(np.radians(120)), 0.25],  # Mic 5
    [0.2 * np.cos(np.radians(240)), 0.2 * np.sin(np.radians(240)), 0.25]  # Mic 6
])

def np_shift(arr, num_shift):
    if num_shift > 0:
        return np.concatenate([np.zeros(num_shift), arr[:-num_shift]])
    elif num_shift < 0:
        return np.concatenate([arr[-num_shift:], np.zeros(-num_shift)])
    else:
        return arr

def beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    ref_mic_position = mic_positions[0]

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            ref_delay = np.dot(ref_mic_position, direction_vector) / c
            delays = (np.dot(mic_positions, direction_vector) / c) - ref_delay

            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np_shift(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1]  # Normalizar por el número de micrófonos
            energy[az_idx, el_idx] = np.sum(output_signal**2)

    return energy

# Configuración de audio
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []
azimuth_range = np.arange(0, 361, 10)
elevation_range = np.arange(0, 91, 10)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energía')
ax.set_xlabel('Azimut (grados)')
ax.set_ylabel('Elevación (grados)')
ax.set_title('Energía del beamforming')

for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    if len(data) % CHANNELS != 0:
        print("Advertencia: El número de muestras no es divisible por el número de canales.")
        continue

    signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

    energy = beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))
    fig.canvas.draw()
    fig.canvas.flush_events()
