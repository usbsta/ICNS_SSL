import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys

# Configuración de audio
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # canales por dispositivo
RATE = 48000  # tasa de muestreo
CHUNK = int(0.2 * RATE)  # tamaño del buffer en 200 ms
d = 0.04  # distancia entre micrófonos
c = 343  # velocidad del sonido en m/s
RECORD_SECONDS = 1200000  # tiempo de grabación
TARGET_DIFFERENCE = 200e-6  # 42 microsegundos, 2 muestras de tiempo
peak_threshold = 1e8  # umbral de pico

# Índices de dispositivos
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

audio = pyaudio.PyAudio()

# Ángulos a analizar
angles_range = np.arange(-90, 91, 1)

# Nombres de archivo para los archivos WAV
output_filenames = ['device_1.wav', 'device_2.wav', 'device_3.wav']

# Preparar gráfico para mostrar en tiempo real
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # Ajustar según rango de energía esperado
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Energía')

# Preparar buffers para almacenar datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]
frames = [[], [], []]  # Listas para almacenar los frames de cada dispositivo

# Variables para almacenar las correcciones de tiempo
correction_12 = None  # Corrección entre dispositivo 1 y 2
correction_13 = None  # Corrección entre dispositivo 2 y 3

# Variables para acumular las correcciones
total_correction_12 = 0
total_correction_13 = 0

# Banderas para sincronizar una sola vez
synced_12 = False
synced_13 = False

# Tamaño de la ventana en muestras (50 ms)
window_size = int(0.05 * RATE)

# Bloqueo para manejar el acceso a los buffers
buffer_lock = threading.Lock()

# Crear evento para detener los hilos de forma segura
stop_event = threading.Event()


def shift_signal(signal, shift_amount):
    if shift_amount > 0:  # Desplazar a la derecha "Retrasando"
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Desplazar a la izquierda "Avanzando"
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal  # No se aplica desplazamiento
    return shifted_signal


def plot_peaks(signals, peaks_indices, labels):
    plt.figure()
    time_axis = np.linspace(0, 50, window_size)  # Eje de tiempo en milisegundos
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None:
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')  # Marcar el pico
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 50 ms Window')
    plt.show()


def record_device(device_index, buffer_index, stop_event):
    global correction_12, correction_13, total_correction_12, total_correction_13

    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)

        print(f"Recording from device {device_index}...")

        while not stop_event.is_set():
            data = stream.read(CHUNK)
            signal_data = np.frombuffer(data, dtype=np.int32)

            with buffer_lock:
                buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

                # Aplica correcciones acumuladas
                if buffer_index == 1 and total_correction_12 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_13)

                frames[buffer_index].append(buffers[buffer_index].tobytes())

    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print(f"Stream closed for device {device_index}")


def stop_recording_and_save():
    stop_event.set()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    print("Stopping recording...")

    # Guardar los datos grabados en archivos WAV
    for i in range(3):
        with wave.open(output_filenames[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames[i]))

    audio.terminate()
    print("Recording saved and audio terminated.")


def signal_handler(sig, frame):
    print("Signal received, stopping recording...")
    stop_recording_and_save()
    sys.exit(0)


# Asignar el manejador de señales para SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)

# Iniciar hilos de grabación
threads = []
threads.append(threading.Thread(target=record_device, args=(device_index_1, 0, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_2, 1, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_3, 2, stop_event)))

# Iniciar los hilos
for thread in threads:
    thread.start()

# Procesamiento en tiempo real
angles_history = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    combined_signal = np.hstack(buffers)
    num_samples = combined_signal.shape[0]
    energy = np.zeros(len(angles_range))

    # Detección de picos en los canales de interés
    peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)
    peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)
    peaks_4, _ = find_peaks(buffers[2][:, 5], height=peak_threshold)

    # Procesar si se detectan picos
    if peaks_1.size > 0 and peaks_2.size > 0 and peaks_4.size > 0:
        # Extraer una ventana de 3 ms alrededor de cada pico para graficar
        start_idx = max(peaks_1[0] - window_size // 2, 0)
        end_idx = start_idx + window_size

        # Asegurarse de que end_idx no exceda la longitud del buffer
        if end_idx > buffers[0].shape[0]:
            end_idx = buffers[0].shape[0]
            start_idx = end_idx - window_size

        # Extraer las señales de los canales relevantes
        signal_1_ch5 = buffers[0][start_idx:end_idx, 5]
        signal_2_ch1 = buffers[1][start_idx:end_idx, 0]
        signal_3_ch1 = buffers[2][start_idx:end_idx, 5]

        # Graficar las señales con los picos detectados
        signals = [signal_1_ch5, signal_2_ch1, signal_3_ch1]
        peaks_indices = [peaks_1[0] - start_idx, peaks_2[0] - start_idx, peaks_4[0] - start_idx]
        labels = ['Device 1 Channel 5', 'Device 2 Channel 1', 'Device 3 Channel 1']
        plot_peaks(signals, peaks_indices, labels)

    if peaks_1.size > 0 and peaks_2.size > 0 and not synced_12:
        peak_time_1 = peaks_1[0] / RATE
        peak_time_2 = peaks_2[0] / RATE
        time_difference_12 = peak_time_2 - peak_time_1
        print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds")

        sample_difference_12 = int(time_difference_12 * RATE)

        if abs(time_difference_12) > TARGET_DIFFERENCE:
            correction_12 = -sample_difference_12
            total_correction_12 += correction_12
            buffers[1] = shift_signal(buffers[1], correction_12)
            print(f"Applying correction of {correction_12} samples to synchronize devices 1 and 2.")
            synced_12 = True  # Establecer la bandera a True después de la primera sincronización

        else:
            print(
                f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds (less than 42 us, no further adjustment)")

    if peaks_1.size > 0 and peaks_4.size > 0 and not synced_13:
        peak_time_1 = peaks_1[0] / RATE
        peak_time_4 = peaks_4[0] / RATE
        time_difference_13 = peak_time_4 - peak_time_1
        print(f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds")

        sample_difference_13 = int(time_difference_13 * RATE)

        if abs(time_difference_13) > TARGET_DIFFERENCE:
            correction_13 = -sample_difference_13
            total_correction_13 += correction_13
            buffers[2] = shift_signal(buffers[2], correction_13)
            print(f"Applying correction of {correction_13} samples to synchronize devices 1 and 3.")
            synced_13 = True  # Establecer la bandera a True después de la primera sincronización

        else:
            print(
                f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds (less than 42 us, no further adjustment)")

    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)
        delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c
        delay_samples = np.round(delays * RATE).astype(int)

        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS * 3):
            delayed_signal = shift_signal(combined_signal[:, i], -delay_samples[i])
            output_signal += delayed_signal
        output_signal /= (CHANNELS * 3)

        energy[idx] = np.sum(output_signal ** 2)

    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)

    # Actualizar el gráfico en tiempo real
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Ajustar dinámicamente el límite del eje y
    fig.canvas.draw()
    fig.canvas.flush_events()

# Detener los hilos de grabación después de que termine el procesamiento
stop_recording_and_save()

if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
