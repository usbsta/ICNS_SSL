#works well
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import threading

FORMAT = pyaudio.paInt32
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 1200000
c = 343
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.6e9


device_index_1 = 3
#device_index_2 = 4  # Zoom 2
#device_index_3 = 5  # Zoom

#output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']

# Posiciones de los micrófonos en 3D
mic_positions = np.array([
    [0.12 * np.cos(np.radians(0)), 0.12 * np.sin(np.radians(0)), -0.25],  # Mic 1
    [0.12 * np.cos(np.radians(120)), 0.12 * np.sin(np.radians(120)), -0.25],  # Mic 2
    [0.12 * np.cos(np.radians(240)), 0.12 * np.sin(np.radians(240)), -0.25],  # Mic 3
    [0.2 * np.cos(np.radians(0)), 0.2 * np.sin(np.radians(0)), -0.5],  # Mic 4
    [0.2 * np.cos(np.radians(120)), 0.2 * np.sin(np.radians(120)), -0.5],  # Mic 5
    [0.2 * np.cos(np.radians(240)), 0.2 * np.sin(np.radians(240)), -0.5]  # Mic 6
])

audio = pyaudio.PyAudio()
#wav_filenames = ['/Users/bjrn/Desktop/Measurements/18ch_3D_outside/device_1_sync.wav']
wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/18ch_3D_outside/device_1_sync.wav']
#wav_filenames = ['device_1_sync.wav']

def skip_wav_seconds(wav_file, seconds, rate, channels, format_size):
    # Calcular el número de frames a saltar
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)

def np_shift(arr, num_shift):
    if num_shift > 0:
        return np.concatenate([np.zeros(num_shift), arr[:-num_shift]])
    elif num_shift < 0:
        return np.concatenate([arr[-num_shift:], np.zeros(-num_shift)])
    else:
        return arr

# Función de beamforming en el dominio del tiempo con micrófono 1 como referencia
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

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

            # Cálculo de los retrasos en tiempo para cada micrófono
            delays = np.dot(mic_positions, direction_vector) / c

            # Aplicar los retrasos alineando las señales
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np_shift(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1]  # Normalizar por el número de micrófonos
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

# Buffers para almacenar los datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(1)]
frames = []
azimuth_range = np.arange(-90, 91, 5)  # Ángulos de azimut de -90° a 90°
azimuth_range = np.arange(-180, 180, 5)
elevation_range = np.arange(-10, 91, 5)  # Ángulos de elevación de -90° a 90°
elevation_range = np.arange(-90, 91, 5)

# Configuración de la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energía')
ax.set_xlabel('Azimut (grados)')
ax.set_ylabel('Elevación (grados)')
ax.set_title('Energía del beamforming')

def record_device(device_index, buffer_index):
    global correction_12, correction_23

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    print(f"Recording from device {device_index}...")

    while True:
        data = stream.read(CHUNK)
        signal_data = np.frombuffer(data, dtype=np.int32)
        buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

        # Almacenar los frames antes de cualquier corrección
        frames[buffer_index].append(buffers[buffer_index].tobytes())

    stream.stop_stream()
    stream.close()

# Función para leer el siguiente bloque de datos desde un archivo WAV (para simulación)
def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))

# Solicitar al usuario el modo de operación
mode = input("Seleccione el numero de modo de operación ('medicion (1)' o 'simulacion (2)'): ").strip().lower()

if mode == '1':
    # Iniciar hilos de grabación en modo medición
    thread_1 = threading.Thread(target=record_device, args=(device_index_1, 0))

    thread_1.start()

elif mode == '2':
    # Abrir los archivos WAV en modo simulación
    wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

    # Tamaño de formato en bytes (esto puede variar, dependiendo de FORMAT)
    format_size = audio.get_sample_size(FORMAT)

    # Saltar los primeros 60 segundos
    skip_seconds = 65

    for wav_file in wav_files:
        skip_wav_seconds(wav_file, skip_seconds, RATE, CHANNELS, format_size)

    try:
        angles_history = []
        for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # Leer el siguiente bloque de datos desde el archivo WAV
            signal_data = read_wav_block(wav_files[0], CHUNK)

            if signal_data is None:
                break  # Salir del bucle si no hay más datos en el archivo WAV

            # Calcular la energía utilizando beamforming en el dominio del tiempo
            energy = beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

            max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
            estimated_azimuth = azimuth_range[max_energy_idx[0]]
            estimated_elevation = elevation_range[max_energy_idx[1]]
            print(f"Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

            # Actualizar los datos del mapa de calor
            cax.set_data(energy.T)
            cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color
            fig.canvas.draw()
            fig.canvas.flush_events()

        print("Simulación completada.")
    finally:
        for wav_file in wav_files:
            wav_file.close()

else:
    print("Modo no reconocido. Seleccione 'medicion' o 'simulacion'.")

audio.terminate()

# Guardar los datos grabados en archivos WAV
output_filenames = ['output_device_1.wav', 'output_device_2.wav', 'output_device_3.wav']
for i in range(3):
    with wave.open(output_filenames[i], 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames[i]))