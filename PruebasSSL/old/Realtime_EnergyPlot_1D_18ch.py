import pyaudio
import numpy as np
from scipy.signal import correlate
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

# Índices de dispositivos
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

audio = pyaudio.PyAudio()

# Angulos a analizar
angles_range = np.arange(-90, 91, 1)

# Preparar gráfico para mostrar en tiempo real
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # Ajustar según rango de energía esperado
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Energía')

# Buffer para almacenar los datos de audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]


def record_device(device_index, buffer_index):
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

    stream.stop_stream()
    stream.close()


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

    num_samples = combined_signal.shape[0]
    energy = np.zeros(len(angles_range))  # energía en cada ángulo

    # Iterar sobre todos los ángulos posibles para encontrar la dirección con máxima energía
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convertir ángulo a radianes
        delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c  # Calcular retardos temporales
        delay_samples = np.round(delays * RATE).astype(int)  # Convertir retardos a muestras

        # Aplicar Beamforming de Retardo y Suma para este ángulo
        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS * 3):
            delayed_signal = np.roll(combined_signal[:, i], -delay_samples[i])
            output_signal += delayed_signal
        output_signal /= (CHANNELS * 3)  # Normalizar

        # Calcular la energía de la señal combinada
        energy[idx] = np.sum(output_signal ** 2)

    # Guardar la energía para cada ángulo y tiempo
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Ángulo estimado en la ventana actual: {estimated_angle:.2f} grados")

    # Actualizar gráfico en tiempo real
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Ajustar dinámicamente el límite del eje y
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Procesamiento de beamforming finalizado.")

# Terminar sesión de audio
audio.terminate()

# Si deseas promediar todos los ángulos calculados
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Ángulo promedio de todas las ventanas: {overall_avg_angle:.2f} grados")
else:
    print("No se pudieron calcular ángulos en ninguna ventana.")
