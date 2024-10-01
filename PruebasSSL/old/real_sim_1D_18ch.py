import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading

# Configuración de audio
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(0.2 * RATE)  # Tamaño del buffer en 200 ms
d = 0.04  # Distancia entre micrófonos
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 900  # Tiempo de grabación
TARGET_DIFFERENCE = 42e-6  # 42 microsegundos, 2 samples time
peak_threshold = 6e8

# Índices de dispositivos
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

audio = pyaudio.PyAudio()

# Ángulos a analizar
angles_range = np.arange(-90, 91, 1)

# Nombres de los archivos WAV (para la opción de simulación)
wav_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']

# Preparar gráfico para mostrar en tiempo real
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # Ajustar según rango de energía esperado
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Energía')

# Buffers para almacenar los datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]
frames = [[], [], []]  # Listas para almacenar los frames de cada dispositivo

# Variables para almacenar las correcciones de tiempo
correction_12 = None  # Corrección entre dispositivo 1 y 2
correction_23 = None  # Corrección entre dispositivo 2 y 3

# Tamaño de la ventana en samples (3 ms)
window_size = int(0.05 * RATE)

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
    time_axis = np.linspace(0, 3, window_size)  # Eje de tiempo en milisegundos
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None:
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')  # Marcar el pico
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 3 ms Window')
    plt.show()

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

        # Aplicar corrección de tiempo en dispositivo 2 si se detecta
        if buffer_index == 1 and correction_12 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_12)

        # Aplicar corrección de tiempo en dispositivo 3 si se detecta
        if buffer_index == 2 and correction_23 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_23)

        # Almacenar los frames después de la corrección
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
mode = input("Seleccione el modo de operación ('medicion' o 'simulacion'): ").strip().lower()

if mode == 'medicion':
    # Iniciar hilos de grabación en modo medición
    thread_1 = threading.Thread(target=record_device, args=(device_index_1, 0))
    thread_2 = threading.Thread(target=record_device, args=(device_index_2, 1))
    thread_3 = threading.Thread(target=record_device, args=(device_index_3, 2))

    thread_1.start()
    thread_2.start()
    thread_3.start()

elif mode == 'simulacion':
    # Abrir los archivos WAV en modo simulación
    wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

    try:
        angles_history = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            # Leer el siguiente bloque de datos para cada dispositivo
            for i, wav_file in enumerate(wav_files):
                block = read_wav_block(wav_file, CHUNK)
                if block is None:
                    break  # Si se alcanzó el final del archivo
                buffers[i] = block
                frames[i].append(block.tobytes())

            combined_signal = np.hstack(buffers)
            num_samples = combined_signal.shape[0]
            energy = np.zeros(len(angles_range))

            # Detección de picos en los canales de interés
            peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)
            peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)
            peaks_3, _ = find_peaks(buffers[1][:, 5], height=peak_threshold)
            peaks_4, _ = find_peaks(buffers[2][:, 0], height=peak_threshold)

            # Procesar si se detectan picos
            if peaks_1.size > 0 and peaks_2.size > 0 and peaks_3.size > 0 and peaks_4.size > 0:
                start_idx = max(peaks_1[0] - window_size // 2, 0)
                end_idx = start_idx + window_size
                if end_idx > buffers[0].shape[0]:
                    end_idx = buffers[0].shape[0]
                    start_idx = end_idx - window_size

                signal_1_ch5 = buffers[0][start_idx:end_idx, 5]
                signal_2_ch1 = buffers[1][start_idx:end_idx, 0]
                signal_2_ch5 = buffers[1][start_idx:end_idx, 5]
                signal_3_ch1 = buffers[2][start_idx:end_idx, 0]

                signals = [signal_1_ch5, signal_2_ch1, signal_2_ch5, signal_3_ch1]
                peaks_indices = [peaks_1[0] - start_idx, peaks_2[0] - start_idx, peaks_3[0] - start_idx, peaks_4[0] - start_idx]
                labels = ['Device 1 Channel 5', 'Device 2 Channel 1', 'Device 2 Channel 5', 'Device 3 Channel 1']
                plot_peaks(signals, peaks_indices, labels)

            if peaks_1.size > 0 and peaks_2.size > 0:
                peak_time_1 = peaks_1[0] / RATE
                peak_time_2 = peaks_2[0] / RATE
                time_difference_12 = peak_time_2 - peak_time_1
                print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds")

                sample_difference_12 = int(time_difference_12 * RATE)

                if abs(time_difference_12) > TARGET_DIFFERENCE:
                    correction_12 = -sample_difference_12
                    buffers[1] = shift_signal(buffers[1], correction_12)
                    print(f"Applying correction of {correction_12} samples to synchronize devices 1 and 2.")
                else:
                    print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds (less than 42 us, no further adjustment)")

            if peaks_3.size > 0 and peaks_4.size > 0:
                peak_time_3 = peaks_3[0] / RATE
                peak_time_4 = peaks_4[0] / RATE
                time_difference_23 = peak_time_4 - peak_time_3
                print(f"Time difference detected between devices 2 and 3: {time_difference_23:.6f} seconds")

                sample_difference_23 = int(time_difference_23 * RATE)

                if abs(time_difference_23) > TARGET_DIFFERENCE:
                    correction_23 = -sample_difference_23
                    buffers[2] = shift_signal(buffers[2], correction_23)
                    print(f"Applying correction of {correction_23} samples to synchronize devices 2 and 3.")
                else:
                    print(f"Time difference detected between devices 2 and 3: {time_difference_23:.6f} seconds (less than 42 us, no further adjustment)")

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
            print(f"Estimated angle in the current window: {estimated_angle:.2f} degrees")

            line.set_ydata(energy)
            ax.set_ylim(0, max(energy) * 1.1)  # Ajustar dinámicamente el límite del eje y
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

if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
