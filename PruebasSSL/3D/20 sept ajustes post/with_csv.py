# no funciona bien
import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import threading
import signal
import sys
from scipy.signal import butter, filtfilt

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(0.2 * RATE)  # Tamaño del buffer en 200 ms
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 3000.0

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(15, 91, 4)

#r = [0.1, 0.15, 0.25, 0.3, 0.4, 0.6]
#r = [0.12, 0.2, 0.3, 0.35, 0.45, 0.65]
#r = [0.22, 0.3, 0.38, 0.43, 0.6, 0.73]
#h = [-1.1, -0.93, -0.77, -0.6, -0.4, -0.01]

#h = [1.1, 0.93, 0.77, 0.6, 0.4, 0.01]
#h = [1.23, 1.06, 0.87, 0.71, 0.53, 0.01]
#h = [0, -0.17, -0.33, -0.5, -0.7, -1.09]
h = [1.1, 0.93, 0.77, 0.6, 0.4, 0.01]
a = [0, 120, 240]
r = [0.12, 0.2, 0.3, 0.35, 0.45, 0.65]

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


# Ajustar el cálculo de azimut y elevación
def calculate_azimuth_and_elevation(lat, lon, alt, mic_lat, mic_lon, mic_height):
    # Convertir la diferencia entre las posiciones del dron y el micrófono
    delta_lat = lat - mic_lat
    delta_lon = lon - mic_lon
    delta_alt = alt - mic_height

    # Distancia horizontal entre el micrófono y el dron en metros (aproximación simple)
    distance_horizontal = np.sqrt(delta_lat**2 + delta_lon**2)

    # Calcular el ángulo de azimut (usamos atan2 para obtener el ángulo en el plano horizontal)
    azimuth = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360

    # Calcular el ángulo de elevación
    elevation = np.degrees(np.arctan2(delta_alt, distance_horizontal))

    return azimuth, elevation

# Posición del micrófono (latitud, longitud, altura en metros)
mic_lat = -33.754454  # Cambia esto según la latitud real del arreglo de micrófonos
mic_lon = 150.74052   # Cambia esto según la longitud real del arreglo de micrófonos
mic_height = 0        # Asumimos que el arreglo de micrófonos está a nivel del suelo (0 metros)


# Leer el archivo CSV
drone_data = pd.read_csv('/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/flight 20 09.csv')

# Variables de sincronización
flight_time = drone_data['OSD.flyTime [s]'].values
latitude = drone_data['OSD.latitude'].values
longitude = drone_data['OSD.longitude'].values
altitude_ft = drone_data['OSD.height [ft]'].values
altitude_m = altitude_ft * 0.3048  # Convertir altura de pies a metros

# Nombres de los archivos WAV (para la opción de simulación)
wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']

wav_filenames = ['/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',
                 '/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']

buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

# beamforming
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

            delays = (np.dot(mic_positions, direction_vector) / c)

            # aplying delays
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1] # normalize amplitud with num of mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)

    return energy

def calculate_time(time_idx, chunk_size, rate):
    # Calcular el tiempo actual en segundos
    time_seconds = (time_idx * chunk_size) / rate
    return time_seconds

def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))


def skip_wav_seconds(wav_file, seconds, rate):
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)

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

plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('beamforming enery')

# Marcador para máxima energía estimada (por beamforming)
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

# Marcador para la posición real del dron (extraída del CSV)
real_position_marker, = ax.plot([], [], 'bo')
real_position_text = ax.text(0, 0, '', color='blue', fontsize=12, ha='center')


wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 51
for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

# ... (todo el código previo permanece igual)

# Lógica de actualización en tiempo real (beamforming y CSV)
# Lógica de actualización en tiempo real (beamforming y CSV)
try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Calcular el tiempo actual en el proceso

        # Leer el siguiente bloque de datos para cada dispositivo
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                break  # Si se alcanzó el final del archivo
            buffers[i] = block

        combined_signal = np.hstack(buffers)

        # filtering
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]
        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Actualizar el ángulo estimado (por beamforming)
        energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        # Encontrar los ángulos reales desde el CSV
        real_time_idx = np.argmin(np.abs(flight_time - current_time))  # Índice más cercano en el tiempo de vuelo
        real_latitude = latitude[real_time_idx]
        real_longitude = longitude[real_time_idx]
        real_altitude = altitude_m[real_time_idx]

        # Calcular azimut y elevación reales basados en las posiciones del dron y el arreglo de micrófonos
        real_azimuth, real_elevation = calculate_azimuth_and_elevation(real_latitude, real_longitude, real_altitude, mic_lat, mic_lon, mic_height)

        # Imprimir los valores de azimut y elevación reales
        print(f"Real Az: {real_azimuth:.1f}°, Real El: {real_elevation:.1f}°")

        # Actualizar la gráfica
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))

        # Actualizar el marcador de la estimación de máxima energía
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        # Actualizar el marcador de la posición real del dron
        real_position_marker.set_data([real_azimuth], [real_elevation])
        real_position_text.set_position((real_azimuth, real_elevation))
        real_position_text.set_text(f"Real Az: {real_azimuth:.1f}°, Real El: {real_elevation:.1f}°")

        # Ajustar los límites de la gráfica si es necesario
        ax.set_xlim([0, 360])  # El azimut debe estar entre 0 y 360 grados
        ax.set_ylim([0, 90])   # La elevación debe estar entre 0 y 90 grados

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Simulación completada.")
finally:
    for wav_file in wav_files:
        wav_file.close()