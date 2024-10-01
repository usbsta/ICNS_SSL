import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuración de parámetros comunes
FORMAT = pyaudio.paInt16  # 16 bits
CHANNELS = 1  # Mono
RATE = 44100  # Frecuencia de muestreo
CHUNK = 2048  # Tamaño del buffer

# Índice del dispositivo
device_index = 0  # Reemplaza con el índice de la Zoom F6

# Inicializar PyAudio
audio = pyaudio.PyAudio()

try:
    # Abrir stream de audio
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    # Crear la figura de matplotlib para la visualización en tiempo real
    fig, ax = plt.subplots()
    x = np.arange(0, 2 * CHUNK, 2)
    line, = ax.plot(x, np.random.rand(CHUNK))

    # Configurar los límites del gráfico
    ax.set_ylim(-32768, 32767)  # Límite para formato paInt16
    ax.set_xlim(0, CHUNK)

    # Función para actualizar la gráfica
    def update(frame):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_np = np.frombuffer(data, dtype=np.int16)
            line.set_ydata(data_np)
        except IOError as e:
            if e.errno == -9981:  # Código de error para "Input overflowed"
                print("Advertencia: Input overflowed")
            else:
                raise
        return line,

    # Crear animación
    ani = animation.FuncAnimation(fig, update, blit=True, interval=100, cache_frame_data=False)

    # Mostrar el gráfico
    plt.show()

finally:
    # Asegurarse de cerrar correctamente el stream y PyAudio
    if stream.is_active():
        stream.stop_stream()
    stream.close()
    audio.terminate()
