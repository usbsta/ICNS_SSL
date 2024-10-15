import numpy as np
import wave
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq

# Buffers para acumular las señales de salida se eliminan

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(1 * RATE)  # Tamaño del buffer en 1 segundo
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 8000.0

# Reducir el número de direcciones
azimuth_range = np.arange(-180, 181, 30)
elevation_range = np.arange(10, 91, 20)

a = [0, -120, -240]
a2 = [-40, -80, -160, -200, -280, -320]

# config 1 equidistance
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

# Nombres de los archivos WAV (asegúrate de poner las rutas correctas)
wav_filenames = [
    '/ruta/al/archivo/device_1_sync.wav',
    '/ruta/al/archivo/device_2_sync.wav',
    '/ruta/al/archivo/device_3_sync.wav'
]

buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(len(wav_filenames))]

# beamforming

def calculate_time(time_idx, chunk_size, rate):
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

# Función optimizada para aplicar retraso fraccionario
def apply_delay(signal, delay, rate):
    N = len(signal)
    freqs = fftfreq(N, d=1/rate)
    signal_fft = fft(signal)
    delay_factor = np.exp(-2j * np.pi * freqs * delay)
    signal_delayed = np.real(ifft(signal_fft * delay_factor))
    return signal_delayed

wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 115

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

# Abrir archivos WAV de salida
max_wav = wave.open('beamformed_max_energy.wav', 'wb')
max_wav.setnchannels(1)
max_wav.setsampwidth(2)
max_wav.setframerate(RATE)

min_wav = wave.open('beamformed_min_energy.wav', 'wb')
min_wav.setnchannels(1)
min_wav.setsampwidth(2)
min_wav.setframerate(RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        finished = False

        # Leer el siguiente bloque de datos para cada dispositivo
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                finished = True
                break
            buffers[i] = block

        if finished:
            print("Fin del archivo de audio.")
            break

        combined_signal = np.hstack(buffers)

        # Aplicar filtro de paso de banda
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]

        energy = np.zeros((len(azimuth_range), len(elevation_range)))

        # Variables para almacenar las señales beamformed de máxima y mínima energía
        output_signal_max = None
        output_signal_min = None

        max_energy = -np.inf
        min_energy = np.inf

        for az_idx, theta in enumerate(azimuth_range):
            azimuth_rad = np.radians(theta)

            for el_idx, phi in enumerate(elevation_range):
                elevation_rad = np.radians(phi)

                # Vector de dirección 3D
                direction_vector = np.array([
                    np.cos(elevation_rad) * np.cos(azimuth_rad),
                    np.cos(elevation_rad) * np.sin(azimuth_rad),
                    np.sin(elevation_rad)
                ])

                delays = (np.dot(mic_positions, direction_vector) / c)

                # Aplicar retrasos fraccionarios
                output_signal = np.zeros(num_samples)
                for mic_idx, delay in enumerate(delays):
                    signal_delayed = apply_delay(filtered_signal[:, mic_idx], delay, RATE)
                    output_signal += signal_delayed

                # Calcular energía
                energy_value = np.sum(output_signal ** 2)
                energy[az_idx, el_idx] = energy_value

                # Verificar si es la energía máxima
                if energy_value > max_energy:
                    max_energy = energy_value
                    output_signal_max = output_signal.copy()
                    max_direction = (theta, phi)

                # Verificar si es la energía mínima
                if energy_value < min_energy:
                    min_energy = energy_value
                    output_signal_min = output_signal.copy()
                    min_direction = (theta, phi)

        # Normalizar y escribir señales de salida
        if output_signal_max is not None:
            max_amp = np.max(np.abs(output_signal_max))
            if max_amp > 0:
                output_signal_max /= max_amp
            output_signal_int16 = np.int16(output_signal_max * 32767)
            max_wav.writeframes(output_signal_int16.tobytes())

        if output_signal_min is not None:
            min_amp = np.max(np.abs(output_signal_min))
            if min_amp > 0:
                output_signal_min /= min_amp
            output_signal_int16 = np.int16(output_signal_min * 32767)
            min_wav.writeframes(output_signal_int16.tobytes())

        # Calcular el tiempo actual
        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Imprimir las direcciones de máxima y mínima energía
        print(f"Tiempo: {current_time + skip_seconds:.2f} s - Máxima energía en Azimut = {max_direction[0]:.2f}°, Elevación = {max_direction[1]:.2f}°")
        print(f"Tiempo: {current_time + skip_seconds:.2f} s - Mínima energía en Azimut = {min_direction[0]:.2f}°, Elevación = {min_direction[1]:.2f}°")

    print("Procesamiento completado.")

finally:
    for wav_file in wav_files:
        wav_file.close()
    max_wav.close()
    min_wav.close()
