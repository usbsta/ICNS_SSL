import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pyproj import Proj, Transformer
import pandas as pd
from numba import njit, prange
from scipy.io import savemat
from scipy.io.wavfile import write

# Definición de parámetros
CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
BUFFER = 0.1  # Tiempo de buffer 100 ms
CHUNK = int(BUFFER * RATE)
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 2000.0

azimuth_range = np.arange(-180, 181, 2)
elevation_range = np.arange(0, 91, 2)

initial_azimuth = -5.0  # Azimut inicial deseado
initial_elevation = 0.0  # Elevación inicial deseada
start_index = 39  # Índice inicial manual del CSV cuando la altura es mayor que 0 metros
skip_seconds = 82  # Tiempo de salto para los archivos WAV.

a = [0, -120, -240]
a2 = [-40, -80, -160, -200, -280, -320]

# Configuración 1 equidistante
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],    # Mic 1  Z3
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],    # Mic 2  Z3
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],    # Mic 3  Z3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],    # Mic 4  Z3
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],    # Mic 5  Z3
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],    # Mic 6  Z3
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],    # Mic 7  Z2
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],    # Mic 8  Z2
    [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],    # Mic 9  Z2
    [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],    # Mic 10 Z2
    [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],    # Mic 11 Z2
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],    # Mic 12 Z2
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],    # Mic 13 Z1
    [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],    # Mic 14 Z1
    [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],    # Mic 15 Z1
    [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],    # Mic 16 Z1
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],    # Mic 17 Z1
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]],    # Mic 18 Z1
    [r[2] * np.cos(np.radians(a2[0])), r[2] * np.sin(np.radians(a2[0])), h[2]],  # Mic 1  Z0
    [r[2] * np.cos(np.radians(a2[1])), r[2] * np.sin(np.radians(a2[1])), h[2]],  # Mic 2  Z0
    [r[2] * np.cos(np.radians(a2[2])), r[2] * np.sin(np.radians(a2[2])), h[2]],  # Mic 3  Z0
    [r[2] * np.cos(np.radians(a2[3])), r[2] * np.sin(np.radians(a2[3])), h[2]],  # Mic 4  Z0
    [r[2] * np.cos(np.radians(a2[4])), r[2] * np.sin(np.radians(a2[4])), h[2]],  # Mic 5  Z0
    [r[2] * np.cos(np.radians(a2[5])), r[2] * np.sin(np.radians(a2[5])), h[2]],  # Mic 6  Z0
])

wav_filenames2 = ['/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_1_nosync.wav',
                  '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_2_nosync.wav',
                  '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_3_nosync.wav',
                  '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_4_nosync.wav']

# Variables para sincronización obtenidas de Sync_Time_Sim_24Ch.py
correction_12 = 866   # Corrección del dispositivo 2
correction_13 = 626   # Corrección del dispositivo 3
correction_14 = 729   # Corrección del dispositivo 4

def shift_signal(signal, shift_amount):
    shifted_signal = np.zeros_like(signal)
    if shift_amount > 0:  # Shift to right (delay)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Shift to left (advance)
        shift_amount = abs(shift_amount)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal.copy()
    return shifted_signal

# Leer los archivos WAV y almacenar los datos
buffers2 = []
params_list = []  # Para almacenar parámetros de los archivos WAV

for idx, wav_file in enumerate(wav_filenames2):
    with wave.open(wav_file, 'rb') as wf:
        # Almacenar parámetros para cada archivo
        params = wf.getparams()
        params_list.append(params)

        # Leer todos los frames
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        # Convertir a arreglo numpy
        signal_data = np.frombuffer(frames, dtype=np.int32)
        # Reestructurar a (num_samples, num_channels)
        signal_data = signal_data.reshape(-1, params.nchannels)
        buffers2.append(signal_data)

# Aplicar desplazamientos a los dispositivos 2, 3 y 4
buffers_shifted = []

# Dispositivo 1 (sin desplazamiento)
buffers_shifted.append(buffers2[0])

# Dispositivo 2
shifted_signal_2 = shift_signal(buffers2[1], correction_12)
buffers_shifted.append(shifted_signal_2)

# Dispositivo 3
shifted_signal_3 = shift_signal(buffers2[2], correction_13)
buffers_shifted.append(shifted_signal_3)

# Dispositivo 4
shifted_signal_4 = shift_signal(buffers2[3], correction_14)
buffers_shifted.append(shifted_signal_4)

# Opcionalmente, guardar las señales desplazadas de nuevo en nuevos archivos WAV
wav_filenames = [
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_1_shifted.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_2_shifted.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_3_shifted.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_4_shifted.wav'
]

for idx, shifted_signal in enumerate(buffers_shifted):
    # Aplanar la señal para escribir en el archivo WAV
    signal_flat = shifted_signal.flatten()
    # Convertir a bytes
    signal_bytes = signal_flat.astype(np.int32).tobytes()
    # Abrir un nuevo archivo WAV para escribir
    with wave.open(wav_filenames[idx], 'wb') as wf:
        # Establecer los parámetros (nchannels, sampwidth, framerate, etc.)
        wf.setparams(params_list[idx])
        # Escribir los frames
        wf.writeframes(signal_bytes)

print("Archivos de audio desplazados han sido guardados.")

# Preparar buffers para almacenar datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(len(wav_filenames))]  # Buffers sincronizados

# Precalcular azimut y elevación en radianes
azimuth_rad = np.radians(azimuth_range)
elevation_rad = np.radians(elevation_range)
num_az = len(azimuth_rad)
num_el = len(elevation_rad)
num_mics = mic_positions.shape[0]

# Crear una cuadrícula de azimut y elevación
az_rad_grid, el_rad_grid = np.meshgrid(azimuth_rad, elevation_rad, indexing='ij')  # Shapes: (num_az, num_el)

# Calcular los vectores de dirección para todas las combinaciones
direction_vectors = np.empty((num_az, num_el, 3), dtype=np.float64)
direction_vectors[:, :, 0] = np.cos(el_rad_grid) * np.cos(az_rad_grid)
direction_vectors[:, :, 1] = np.cos(el_rad_grid) * np.sin(az_rad_grid)
direction_vectors[:, :, 2] = np.sin(el_rad_grid)

# Expandir mic_positions para broadcasting
mic_positions_expanded = mic_positions[:, np.newaxis, np.newaxis, :]  # Shape: (num_mics, 1, 1, 3)
direction_vectors_expanded = direction_vectors[np.newaxis, :, :, :]  # Shape: (1, num_az, num_el, 3)

# Calcular los retrasos
delays = np.sum(mic_positions_expanded * direction_vectors_expanded, axis=3) / c  # Shape: (num_mics, num_az, num_el)

# Calcular delay_samples
delay_samples = np.round(delays * RATE).astype(np.int32)  # Shape: (num_mics, num_az, num_el)

# Funciones de beamforming
@njit(parallel=True)
def beamform_time(signal_data, delay_samples):
    num_samples, num_mics = signal_data.shape
    num_mics, num_az, num_el = delay_samples.shape
    energy = np.zeros((num_az, num_el))

    for az_idx in prange(num_az):
        for el_idx in range(num_el):
            output_signal = np.zeros(num_samples)
            for mic_idx in range(num_mics):
                delay = delay_samples[mic_idx, az_idx, el_idx]
                shifted_signal = shift_signal(signal_data[:, mic_idx], delay)
                output_signal += shifted_signal

            output_signal /= num_mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

@njit
def shift_signal(signal, delay_samples):
    num_samples = signal.shape[0]
    shifted_signal = np.zeros_like(signal)

    if delay_samples > 0:
        # Desplazar hacia adelante (retraso), rellenar al inicio
        if delay_samples < num_samples:
            shifted_signal[delay_samples:] = signal[:-delay_samples]
    elif delay_samples < 0:
        # Desplazar hacia atrás (adelanto), rellenar al final
        delay_samples = -delay_samples
        if delay_samples < num_samples:
            shifted_signal[:-delay_samples] = signal[delay_samples:]
    else:
        # Sin desplazamiento
        shifted_signal = signal.copy()

    return shifted_signal

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

# Diseño del filtro pasa banda
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

######          CSV         ########

# Cargar los archivos CSV
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-10-11_[14-32-34].csv'
file_path_flight = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-10-11_[15-49-12].csv'

# Leer el archivo de referencia y de vuelo
ref_data = pd.read_csv(ref_file_path, skiprows=1, delimiter=',', low_memory=False)
flight_data = pd.read_csv(file_path_flight, skiprows=1, delimiter=',', low_memory=False)

# Extraer la posición de referencia (promedio de los valores válidos de latitud y longitud)
reference_latitude = ref_data['OSD.latitude'].dropna().astype(float).mean()
reference_longitude = ref_data['OSD.longitude'].dropna().astype(float).mean()

# Extraer las columnas necesarias: latitud, longitud, altura y tiempo
latitude_col = 'OSD.latitude'
longitude_col = 'OSD.longitude'
altitude_col = 'OSD.altitude [ft]'
time_col = 'OSD.flyTime'

# Reajustar el dataframe para empezar desde el índice deseado
flight_data = flight_data.iloc[start_index:].reset_index(drop=True)

# Filtrar las filas con datos válidos
flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()

# Convertir la altitud de pies a metros
flight_data[altitude_col] = flight_data[altitude_col] * 0.3048

# Obtener la altitud inicial del vuelo para usarla en la referencia de elevación
initial_altitude = flight_data[altitude_col].iloc[0]

# Configurar la proyección UTM para convertir las coordenadas geográficas a metros
transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)  # Ajustar la zona UTM 56 south

# Convertir las coordenadas de referencia a metros
ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

# Crear nuevas columnas para las coordenadas en metros en el dataframe de vuelo
flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
    flight_data[longitude_col].values,
    flight_data[latitude_col].values
)

# Crear un DataFrame vacío para almacenar los datos
data_columns = ['Tiempo_Audio', 'Tiempo_CSV', 'Azimut_Estimado', 'Elevacion_Estimada',
                'Azimut_CSV', 'Elevacion_CSV', 'Dif_Azimut', 'Dif_Elevacion', 'Distancia_Metros']
results_df = pd.DataFrame(columns=data_columns)

# Función para calcular la distancia horizontal entre dos puntos en coordenadas cartesianas
def calculate_horizontal_distance_meters(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def wrap_angle(angle):
    return ((angle + 180) % 360) - 180

# Función para calcular el azimut usando coordenadas en metros
def calculate_azimuth_meters(x1, y1, x2, y2):
    azimuth = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return azimuth

# Función para calcular la elevación utilizando la distancia horizontal en metros
def calculate_elevation_meters(altitude, x1, y1, x2, y2, reference_altitude):
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    relative_altitude = altitude - reference_altitude  # Calcular la altura relativa respecto a la referencia
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

# Función para calcular la distancia total (hipotenusa) entre dos puntos, considerando la altitud
def calculate_total_distance_meters(x1, y1, x2, y2, alt1, alt2):
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    altitude_difference = alt2 - alt1
    total_distance = np.sqrt(horizontal_distance**2 + altitude_difference**2)
    return total_distance

def angular_difference(angle1, angle2):
    diff = (angle1 - angle2 + 180) % 360 - 180
    return abs(diff)

def calculate_angle_difference(beamform_az, csv_az, beamform_el, csv_el):
    az_diff = angular_difference(beamform_az, csv_az)
    el_diff = abs(beamform_el - csv_el)
    return az_diff, el_diff

# Calcular el azimut y elevación inicial reales del dron
drone_initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                                 flight_data.iloc[0]['X_meters'],
                                                 flight_data.iloc[0]['Y_meters'])
drone_initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col],
                                                    ref_x, ref_y,
                                                    flight_data.iloc[0]['X_meters'],
                                                    flight_data.iloc[0]['Y_meters'],
                                                    initial_altitude)

# Calcular los offsets necesarios
azimuth_offset = initial_azimuth - drone_initial_azimuth
elevation_offset = initial_elevation - drone_initial_elevation

# Función update modificada
def update(frame):
    x = flight_data.iloc[frame]['X_meters']
    y = flight_data.iloc[frame]['Y_meters']
    altitude = flight_data.iloc[frame][altitude_col]

    # Calcular el azimut ajustado
    csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + azimuth_offset
    csv_azimuth = wrap_angle(csv_azimuth)
    csv_azimuth = -csv_azimuth

    # Calcular la elevación ajustada
    csv_elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) + elevation_offset

    # Calcular la distancia total
    total_distance = calculate_total_distance_meters(ref_x, ref_y, x, y, initial_altitude, altitude)

    # Actualizar la posición del punto azul en el gráfico
    point.set_data([csv_azimuth], [csv_elevation])

    return csv_azimuth, csv_elevation, total_distance

########            ########

# Configuración inicial de la visualización
plt.ion()
fig, ax = plt.subplots(figsize=(15, 5))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                origin='lower', aspect='auto', cmap='viridis')

# Punto que representa la posición del dron
point, = ax.plot([], [], 'bo', markersize=5)  # Crear el punto azul
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevación')
ax.set_title('Energía del Beamforming')

# Marcador de la máxima energía
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

# Generar 50 posiciones aleatorias dentro de los rangos válidos
random_azimuths = np.random.uniform(-180, 180, 50)
random_elevations = np.random.uniform(0, 90, 50)

# Encontrar los índices más cercanos en los rangos de azimut y elevación
random_az_indices = np.array([np.argmin(np.abs(azimuth_range - az)) for az in random_azimuths])
random_el_indices = np.array([np.argmin(np.abs(elevation_range - el)) for el in random_elevations])

# Calcular el número total de iteraciones
num_iterations = min(int(RATE / CHUNK * RECORD_SECONDS), len(flight_data))

# Modificación: Preasignación de arrays para evitar uso de append
beamformed_signal_array = np.zeros((num_iterations, CHUNK))
beamformed_signal_up_array = np.zeros((num_iterations, CHUNK))
beamformed_signal_down_array = np.zeros((num_iterations, CHUNK))
beamformed_signal_right_array = np.zeros((num_iterations, CHUNK))
beamformed_signal_left_array = np.zeros((num_iterations, CHUNK))
beamformed_signal_random_arrays = np.zeros((50, num_iterations, CHUNK))

# Lista para almacenar las matrices de energía en cada tiempo
energy_data = []

# Loop principal (para procesamiento y almacenamiento de datos)
try:
    for time_idx, i in zip(range(0, num_iterations), range(len(flight_data))):
        finished = False

        # Leer el siguiente bloque de datos para cada dispositivo
        for j, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                finished = True  # Set flag to break out of the loop
                break  # Si se alcanzó el final del archivo
            buffers[j] = block

        if finished:
            print("Fin del archivo de audio.")
            break

        combined_signal = np.hstack(buffers)

        # Actualizar la posición del dron y obtener los ángulos y la distancia del CSV
        csv_azimuth, csv_elevation, total_distance = update(i)

        # Encontrar los índices más cercanos en los rangos de azimut y elevación
        az_idx = (np.abs(azimuth_range - csv_azimuth)).argmin()
        el_idx = (np.abs(elevation_range - csv_elevation)).argmin()

        # Obtener los retrasos para la posición actual del dron
        delays_at_drone = delay_samples[:, az_idx, el_idx]
        beamformed_signal_chunk = np.zeros(combined_signal.shape[0])
        for mic_idx in range(num_mics):
            delay = delays_at_drone[mic_idx]
            shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
            beamformed_signal_chunk += shifted_signal

        beamformed_signal_chunk /= num_mics  # Normalizar

        # Modificación: Almacenar el chunk en el array preasignado
        beamformed_signal_array[time_idx] = beamformed_signal_chunk

        # Calcular posiciones desplazadas y generar audios correspondientes

        # Elevación +10 grados
        new_elevation_up = csv_elevation + 5
        if new_elevation_up <= elevation_range[-1]:
            el_idx_up = (np.abs(elevation_range - new_elevation_up)).argmin()
            delays_up = delay_samples[:, az_idx, el_idx_up]
            beamformed_signal_chunk_up = np.zeros(combined_signal.shape[0])
            for mic_idx in range(num_mics):
                delay = delays_up[mic_idx]
                shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
                beamformed_signal_chunk_up += shifted_signal
            beamformed_signal_chunk_up /= num_mics
            # Modificación: Almacenar el chunk en el array preasignado
            beamformed_signal_up_array[time_idx] = beamformed_signal_chunk_up

        # Elevación -10 grados
        new_elevation_down = csv_elevation - 5
        if new_elevation_down >= elevation_range[0]:
            el_idx_down = (np.abs(elevation_range - new_elevation_down)).argmin()
            delays_down = delay_samples[:, az_idx, el_idx_down]
            beamformed_signal_chunk_down = np.zeros(combined_signal.shape[0])
            for mic_idx in range(num_mics):
                delay = delays_down[mic_idx]
                shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
                beamformed_signal_chunk_down += shifted_signal
            beamformed_signal_chunk_down /= num_mics
            # Modificación: Almacenar el chunk en el array preasignado
            beamformed_signal_down_array[time_idx] = beamformed_signal_chunk_down

        # Azimut +10 grados (derecha)
        new_azimuth_right = csv_azimuth + 5
        if new_azimuth_right > 180:
            new_azimuth_right -= 360
        az_idx_right = (np.abs(azimuth_range - new_azimuth_right)).argmin()
        delays_right = delay_samples[:, az_idx_right, el_idx]
        beamformed_signal_chunk_right = np.zeros(combined_signal.shape[0])
        for mic_idx in range(num_mics):
            delay = delays_right[mic_idx]
            shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
            beamformed_signal_chunk_right += shifted_signal
        beamformed_signal_chunk_right /= num_mics
        # Modificación: Almacenar el chunk en el array preasignado
        beamformed_signal_right_array[time_idx] = beamformed_signal_chunk_right

        # Azimut -10 grados (izquierda)
        new_azimuth_left = csv_azimuth - 5
        if new_azimuth_left < -180:
            new_azimuth_left += 360
        az_idx_left = (np.abs(azimuth_range - new_azimuth_left)).argmin()
        delays_left = delay_samples[:, az_idx_left, el_idx]
        beamformed_signal_chunk_left = np.zeros(combined_signal.shape[0])
        for mic_idx in range(num_mics):
            delay = delays_left[mic_idx]
            shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
            beamformed_signal_chunk_left += shifted_signal
        beamformed_signal_chunk_left /= num_mics
        # Modificación: Almacenar el chunk en el array preasignado
        beamformed_signal_left_array[time_idx] = beamformed_signal_chunk_left

        # Posiciones aleatorias
        for idx_random in range(50):
            az_idx_rand = random_az_indices[idx_random]
            el_idx_rand = random_el_indices[idx_random]
            delays_rand = delay_samples[:, az_idx_rand, el_idx_rand]
            beamformed_signal_chunk_rand = np.zeros(combined_signal.shape[0])
            for mic_idx in range(num_mics):
                delay = delays_rand[mic_idx]
                shifted_signal = shift_signal(combined_signal[:, mic_idx], delay)
                beamformed_signal_chunk_rand += shifted_signal
            beamformed_signal_chunk_rand /= num_mics
            # Modificación: Almacenar el chunk en el array preasignado
            beamformed_signal_random_arrays[idx_random, time_idx] = beamformed_signal_chunk_rand

        # Filtrar la señal
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

        # energy = beamform_frequency(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
        energy = beamform_time(filtered_signal, delay_samples)
        energy_data.append(energy)

        # Encontrar el índice de la máxima energía
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        # Calcular el tiempo actual de la muestra de audio
        current_time_audio = calculate_time(time_idx, CHUNK, RATE)

        # Calcular la diferencia entre los ángulos del beamforming y el CSV
        azimuth_diff, elevation_diff = calculate_angle_difference(estimated_azimuth, csv_azimuth, estimated_elevation, csv_elevation)

        # Obtener el tiempo del CSV (tiempo de vuelo)
        current_time_csv = flight_data.iloc[i][time_col]

        # Crear un nuevo DataFrame con los datos actuales
        new_data = pd.DataFrame([{
            'Tiempo_Audio': current_time_audio + skip_seconds,
            'Tiempo_CSV': current_time_csv,
            'Azimut_Estimado': estimated_azimuth,
            'Elevacion_Estimada': estimated_elevation,
            'Azimut_CSV': csv_azimuth,
            'Elevacion_CSV': csv_elevation,
            'Dif_Azimut': azimuth_diff,
            'Dif_Elevacion': elevation_diff,
            'Distancia_Metros': total_distance
        }])

        # Concatenar el nuevo DataFrame con el existente
        results_df = pd.concat([results_df, new_data], ignore_index=True)

        # Imprimir los datos para monitoreo
        print(f"Audio time: {current_time_audio + skip_seconds:.2f} s - CSV time: {current_time_csv} s - " \
              f"SSL: Azim = {estimated_azimuth:.2f}°, Elev = {estimated_elevation:.2f}° " \
              f"CSV: Azim = {csv_azimuth:.2f}°, Elev = {csv_elevation:.2f}° " \
              f"Diff: Azim = {azimuth_diff:.2f}°, Elev = {elevation_diff:.2f}° - " \
              f"Dist: {total_distance:.2f} mts")

        # Actualizar la posición del marcador de máxima energía
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Actualizar la posición del texto con las coordenadas
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        # Actualizar los datos del mapa de calor
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color

        #fig.canvas.draw()
        #fig.canvas.flush_events()

    # Guardar la lista de matrices de energía en un archivo .mat al final del bucle
    # savemat('energy_data.mat', {'energy_data': energy_data})
    print("Simulación completada.")
    plt.ioff()
    plt.show()

finally:
    # Asegurarse de cerrar los archivos wav
    for wav_file in wav_files:
        wav_file.close()

    # Guardar los resultados en un archivo CSV para análisis posterior
    print(f"Guardando archivo CSV. Total de filas: {len(results_df)}")
    results_df.to_csv('beamforming_results.csv', index=False)
    print("Datos guardados en 'beamforming_results.csv'.")

    # Después del bucle, concatenar y guardar las señales beamformed
    # Función para normalizar y guardar la señal
    def save_beamformed_signal(beamformed_signal_array, filename):
        beamformed_signal = beamformed_signal_array.flatten()
        # Normalizar la señal beamformed
        max_abs_value = np.max(np.abs(beamformed_signal))
        if max_abs_value > 0:
            beamformed_signal /= max_abs_value  # Normalizar entre -1 y 1
        # Convertir a formato int16
        beamformed_signal_int16 = np.int16(beamformed_signal * 32767)
        # Guardar en archivo WAV
        write(filename, RATE, beamformed_signal_int16)
        print(f"Señal beamformed guardada en '{filename}'.")

    # Guardar las señales en las posiciones solicitadas
    save_beamformed_signal(beamformed_signal_array, 'beamformed_signal_csv_realpos.wav')
    save_beamformed_signal(beamformed_signal_up_array, 'beamformed_signal_5deg_up.wav')
    save_beamformed_signal(beamformed_signal_down_array, 'beamformed_signal_5deg_down.wav')
    save_beamformed_signal(beamformed_signal_right_array, 'beamformed_signal_5deg_right.wav')
    save_beamformed_signal(beamformed_signal_left_array, 'beamformed_signal_5deg_left.wav')

    # Guardar las señales de las posiciones aleatorias
    for idx_random in range(50):
        beamformed_signal_random = beamformed_signal_random_arrays[idx_random].flatten()
        # Normalizar la señal beamformed
        max_abs_value = np.max(np.abs(beamformed_signal_random))
        if max_abs_value > 0:
            beamformed_signal_random /= max_abs_value  # Normalizar entre -1 y 1
        # Convertir a formato int16
        beamformed_signal_random_int16 = np.int16(beamformed_signal_random * 32767)
        # Obtener los ángulos para nombrar el archivo
        az = random_azimuths[idx_random]
        el = random_elevations[idx_random]
        filename = f'beamformed_signal_random_{idx_random}_az{az:.1f}_el{el:.1f}.wav'
        # Guardar en archivo WAV
        write(filename, RATE, beamformed_signal_random_int16)
        print(f"Señal beamformed aleatoria guardada en '{filename}'.")

    print("Todas las señales beamformed han sido guardadas.")
