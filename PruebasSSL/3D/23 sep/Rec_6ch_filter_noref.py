# works well
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Configuración del audio
FORMAT = pyaudio.paInt32
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 120000
c = 343

lowcut = 400.0
highcut = 8000.0
a = [0, 120, 240]

# zoom 3 up array
device_index = 3
h = [1.12, 0.92]
r = [0.1, 0.17]
output_filename = '6Ch_device_1_sync.wav'

"""
# zoom 2 mid arra
device_index = 4
h = [0.77, 0.6]
r = [0.25, 0.32]
output_filename = '6Ch_device_2_sync.wav'

# zoom 1 down arra
device_index = 1
h = [0.42, 0.2]
r = [0.42, 0.62]
output_filename = '6Ch_device_2_sync.wav'
"""

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]]
])

# Función de beamforming
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
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
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1]
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

# Filtros de paso de banda
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal

# Inicialización de audio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []
azimuth_range = np.arange(-180, 180, 5)
elevation_range = np.arange(10, 91, 5)

# Configuración de la gráfica
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('Beamforming Energy')

max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

        filtered_signal = apply_bandpass_filter(signal_data, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]

        energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
        #energy = beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        print(f"Estimated angle: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Grabación detenida.")

finally:
    # Guardar el archivo de audio
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Archivo de audio guardado como {output_filename}")

# Cerrar el stream y el objeto PyAudio
stream.stop_stream()
stream.close()
audio.terminate()

"""
#works well but old, does not record when stop manually"

import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

FORMAT = pyaudio.paInt32
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 1200000
c = 343
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.6e9


device_index = 3

output_filename = ['device_1_sync.wav']

lowcut = 400.0
highcut = 8000.0

a = [0, 120, 240]

#config 1 equi
h = [1.12, 0.92]
r = [0.1, 0.17]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]]  # Mic 6
])

def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # 3D direction vector
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            delays = np.dot(mic_positions, direction_vector) / c

            # aplying delays
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                #signal_shifted = np_shift(signal_data[:, i], delay_samples)
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1] # normalize amplitud with num of mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

# band pass design
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)  # Aplicar filtro a lo largo de la señal en cada canal
    return filtered_signal

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
azimuth_range = np.arange(-180, 180, 5)
elevation_range = np.arange(5, 91, 5)

# Configuración de la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('beamforming enery')

# max energy point
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')


for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convertir los datos binarios a arreglos numpy
    signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

    #filtered_signal = apply_bandpass_filter(signal_data, lowcut, highcut, RATE)
    #num_samples = filtered_signal.shape[0]

    #energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
    energy = beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Estimated angle: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")
    #print(np.min(energy),np.max(energy))

    # Actualizar los datos del mapa de calor
    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color

    # Actualizar la posición del marcador de máxima energía
    max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

    # Actualizar la posición del texto con las coordenadas
    max_energy_text.set_position((estimated_azimuth, estimated_elevation))
    max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

    fig.canvas.draw()
    fig.canvas.flush_events()


# Guardar el archivo de audio
wf = wave.open(output_filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Archivo de audio guardado como {output_filename}")
"""

