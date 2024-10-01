import pyaudio
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading

# Configuración de audio
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # canales por dispositivo
RATE = 48000  # frecuencia de muestreo
CHUNK = int(0.2 * RATE)  # tamaño del buffer en 200 ms
d = 0.04  # distancia entre micrófonos
c = 343  # velocidad del sonido en m/s
RECORD_SECONDS = 900  # tiempo de grabación
TARGET_DIFFERENCE = 42e-6  # 42 microsegundos

# Índices de dispositivos
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

audio = pyaudio.PyAudio()

# Buffer para almacenar los datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

# Umbral para detección de picos (ajustar según sea necesario)
peak_threshold = 1e8

# Variables para almacenar las correcciones de tiempo que se aplicarán de manera constante
correction_12 = None  # Corrección entre dispositivo 1 y 2
correction_23 = None  # Corrección entre dispositivo 2 y 3

# Variables para almacenar el menor tiempo de ajuste
min_time_difference_12 = None
min_time_difference_23 = None

def shift_signal(signal, shift_amount):
    """Desplaza el array de señal hacia la derecha o izquierda llenando con ceros"""
    if shift_amount > 0:  # Desplazamiento hacia la derecha
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Desplazamiento hacia la izquierda
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal  # No se aplica desplazamiento
    return shifted_signal

def record_device(device_index, buffer_index):
    global correction_12, correction_23

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    print(f"Grabando desde dispositivo {device_index}...")

    while True:
        data = stream.read(CHUNK)
        signal_data = np.frombuffer(data, dtype=np.int32)
        buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

        # Aplicar corrección de tiempo constante en el dispositivo 2 si se ha detectado
        if buffer_index == 1 and correction_12 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_12)

        # Aplicar corrección de tiempo constante en el dispositivo 3 si se ha detectado
        if buffer_index == 2 and correction_23 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_23)

    stream.stop_stream()
    stream.close()

def plot_peak_signals(buffers, peaks_1, peaks_2, peaks_3, peaks_4):
    # Duración de la ventana en muestras
    window_samples = int(3e-3 * RATE)

    # Extraer las señales relevantes de los buffers
    signal_1 = buffers[0][:, 5]  # Device 1, Channel 5
    signal_2_1 = buffers[1][:, 0]  # Device 2, Channel 1
    signal_2_5 = buffers[1][:, 5]  # Device 2, Channel 5
    signal_3_1 = buffers[2][:, 0]  # Device 3, Channel 1

    # Crear un vector de tiempo para el eje x
    time_vector = np.arange(-window_samples, window_samples) / RATE * 1e3  # En milisegundos

    # Preparar la figura y subplots
    plt.figure()

    # Gráfico para dispositivo 1, canal 5
    plt.subplot(4, 1, 1)
    peak_1 = peaks_1[0]
    plt.plot(time_vector, signal_1[peak_1 - window_samples:peak_1 + window_samples])
    plt.title("Device 1, Channel 5")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Gráfico para dispositivo 2, canal 1
    plt.subplot(4, 1, 2)
    peak_2 = peaks_2[0]
    plt.plot(time_vector, signal_2_1[peak_2 - window_samples:peak_2 + window_samples])
    plt.title("Device 2, Channel 1")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Gráfico para dispositivo 2, canal 5
    plt.subplot(4, 1, 3)
    peak_3 = peaks_3[0]
    plt.plot(time_vector, signal_2_5[peak_3 - window_samples:peak_3 + window_samples])
    plt.title("Device 2, Channel 5")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Gráfico para dispositivo 3, canal 1
    plt.subplot(4, 1, 4)
    peak_4 = peaks_4[0]
    plt.plot(time_vector, signal_3_1[peak_4 - window_samples:peak_4 + window_samples])
    plt.title("Device 3, Channel 1")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")

    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    plt.show()

# Iniciar threads de grabación
thread_1 = threading.Thread(target=record_device, args=(device_index_1, 0))
thread_2 = threading.Thread(target=record_device, args=(device_index_2, 1))
thread_3 = threading.Thread(target=record_device, args=(device_index_3, 2))

thread_1.start()
thread_2.start()
thread_3.start()

# Procesamiento en tiempo real
angles_history = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    # Combinar los datos de los tres dispositivos
    combined_signal = np.hstack(buffers)

    # Detección de picos en los canales de interés
    peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)  # Canal 6 del dispositivo 1
    peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)  # Canal 1 del dispositivo 2
    peaks_3, _ = find_peaks(buffers[1][:, 5], height=peak_threshold)  # Canal 6 del dispositivo 2
    peaks_4, _ = find_peaks(buffers[2][:, 0], height=peak_threshold)  # Canal 1 del dispositivo 3

    # Calcular la diferencia de tiempo entre los picos detectados para dispositivos 1 y 2
    if peaks_1.size > 0 and peaks_2.size > 0:
        peak_time_1 = peaks_1[0] / RATE  # Tiempo del primer pico en el canal 6 del dispositivo 1
        peak_time_2 = peaks_2[0] / RATE  # Tiempo del primer pico en el canal 1 del dispositivo 2
        time_difference_12 = peak_time_2 - peak_time_1
        print(f"Diferencia de tiempo detectada entre dispositivos 1 y 2: {time_difference_12:.6f} segundos")

        # Actualizar la menor diferencia de tiempo encontrada y aplicar corrección
        if min_time_difference_12 is None or abs(time_difference_12) < abs(min_time_difference_12):
            min_time_difference_12 = time_difference_12
            sample_difference_12 = int(time_difference_12 * RATE)
            correction_12 = -sample_difference_12
            print(f"Aplicando nueva corrección mínima entre dispositivos 1 y 2: {correction_12} muestras.")

    # Calcular la diferencia de tiempo entre los picos detectados para dispositivos 2 y 3
    if peaks_3.size > 0 and peaks_4.size > 0:
        peak_time_3 = peaks_3[0] / RATE  # Tiempo del primer pico en el canal 6 del dispositivo 2
        peak_time_4 = peaks_4[0] / RATE  # Tiempo del primer pico en el canal 1 del dispositivo 3
        time_difference_23 = peak_time_4 - peak_time_3
        print(f"Diferencia de tiempo detectada entre dispositivos 2 y 3: {time_difference_23:.6f} segundos")

        # Actualizar la menor diferencia de tiempo encontrada y aplicar corrección
        if min_time_difference_23 is None or abs(time_difference_23) < abs(min_time_difference_23):
            min_time_difference_23 = time_difference_23
            sample_difference_23 = int(time_difference_23 * RATE)
            correction_23 = -sample_difference_23
            print(f"Aplicando nueva corrección mínima entre dispositivos 2 y 3: {correction_23} muestras.")

    # Graficar los picos en una ventana de 3 ms
    if peaks_1.size > 0 and peaks_2.size > 0 and peaks_3.size > 0 and peaks_4.size > 0:
        plot_peak_signals(buffers, peaks_1, peaks_2, peaks_3, peaks_4)

# Terminar sesión de audio
audio.terminate()

print("Procesamiento de beamforming finalizado.")
