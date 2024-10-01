#works well

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


device_index = 3

output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']

h = [0, -0.25]
h = [0.5, 0.25]
a = [0, 120, 240]
r = [0.12, 0.2]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]]  # Mic 6
])

# Función de beamforming en el dominio del tiempo con micrófono 1 como referencia
def beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    # Micrófono de referencia es el micrófono 1, su posición es mic_positions[0]
    ref_mic_position = mic_positions[0]

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # Vector de dirección en 3D
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            # Cálculo de los retrasos en tiempo para cada micrófono con respecto al micrófono 1
            ref_delay = np.dot(ref_mic_position, direction_vector) / c
            delays = (np.dot(mic_positions, direction_vector) / c) - ref_delay

            # Aplicar los retrasos alineando las señales
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))  # Convertir el retraso en muestras
                if delay_samples > 0:
                    output_signal[:num_samples - delay_samples] += signal_data[delay_samples:, i]

                else:
                    output_signal[-delay_samples:] += signal_data[:num_samples + delay_samples, i]


            # Cálculo de la energía total
            output_signal /= signal_data.shape[1]  # Normalizar por el número de micrófonos
            energy[az_idx, el_idx] = np.sum(output_signal**2)

    return energy

# Configuración de audio
audio = pyaudio.PyAudio()

# Crear el objeto de stream para grabar
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []
azimuth_range = np.arange(-90, 91, 5)  # Ángulos de azimut de -90° a 90°
elevation_range = np.arange(-10, 91, 5)  # Ángulos de elevación de -90° a 90°
elevation_range = np.arange(-90, 91, 5)

# Configuración de la gráfica en tiempo real
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

    # Convertir los datos binarios a arreglos numpy
    signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

    # Calcular la energía utilizando beamforming en el dominio del tiempo
    energy = beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")
    #print(np.min(energy),np.max(energy))

    # Actualizar los datos del mapa de calor
    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color
    #cax.set_clim(vmin=2.5e18, vmax=3e18)
    fig.canvas.draw()
    fig.canvas.flush_events()

