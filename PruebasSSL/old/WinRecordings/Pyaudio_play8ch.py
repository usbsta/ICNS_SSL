import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del audio
file_path = 'DREGON_free-flight_speech-high_room1.wav'
buffer_duration = 0.2  # 200 ms

# Abrir el archivo WAV
wav_file = wave.open(file_path, 'rb')
n_channels = wav_file.getnchannels()
samplerate = wav_file.getframerate()

# Calcular el tamaño del buffer
buffer_size = int(buffer_duration * samplerate)

# Configuración de pyaudio
p = pyaudio.PyAudio()

# Abrir flujo de audio
stream = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                channels=n_channels,
                rate=samplerate,
                output=True)

# Configuración del plot
fig, axes = plt.subplots(nrows=n_channels, ncols=1, figsize=(10, 8))
lines = []
time = np.linspace(0, buffer_duration, num=buffer_size)

for i in range(n_channels):
    line, = axes[i].plot(time, np.zeros(buffer_size))
    axes[i].set_ylim([-32768, 32767])
    axes[i].set_xlim([0, buffer_duration])
    axes[i].set_title(f'Canal {i + 1}')
    axes[i].set_xlabel('Tiempo [s]')
    axes[i].set_ylabel('Amplitud')
    lines.append(line)

plt.tight_layout()
plt.ion()  # Modo interactivo de matplotlib
plt.show()


# Función para actualizar el plot
def update_plot(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_data = np.reshape(audio_data, (-1, n_channels))

    # Verificar si la longitud del buffer es la misma que la esperada
    if audio_data.shape[0] != buffer_size:
        return  # Saltar la actualización si los tamaños no coinciden

    for i in range(n_channels):
        lines[i].set_ydata(audio_data[:, i])
    fig.canvas.draw()
    fig.canvas.flush_events()


# Leer y reproducir el audio en bloques
data = wav_file.readframes(buffer_size)
while len(data) > 0:
    stream.write(data)
    update_plot(data)
    data = wav_file.readframes(buffer_size)

# Cerrar el flujo y liberar recursos
stream.stop_stream()
stream.close()
p.terminate()
wav_file.close()

"""
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del audio
file_path = 'DREGON_free-flight_speech-high_room1.wav'
buffer_duration = 0.2  # 200 ms

# Abrir el archivo WAV
wav_file = wave.open(file_path, 'rb')
n_channels = wav_file.getnchannels()
samplerate = wav_file.getframerate()

# Calcular el tamaño del buffer
buffer_size = int(buffer_duration * samplerate)

# Configuración de pyaudio
p = pyaudio.PyAudio()

# Abrir flujo de audio
stream = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                channels=n_channels,
                rate=samplerate,
                output=True)

# Configuración del plot
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 8))
lines = []
time = np.linspace(0, buffer_duration, num=buffer_size)

for i in range(8):
    line, = axes[i].plot(time, np.zeros(buffer_size))
    axes[i].set_ylim([-32768, 32767])
    axes[i].set_xlim([0, buffer_duration])
    axes[i].set_title(f'Canal {i+1}')
    axes[i].set_xlabel('Tiempo [s]')
    axes[i].set_ylabel('Amplitud')
    lines.append(line)

plt.tight_layout()
plt.ion()  # Modo interactivo de matplotlib
plt.show()

# Función para actualizar el plot
def update_plot(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_data = np.reshape(audio_data, (-1, n_channels))
    for i in range(8):
        lines[i].set_ydata(audio_data[:, i])
    fig.canvas.draw()
    fig.canvas.flush_events()

# Leer y reproducir el audio en bloques
data = wav_file.readframes(buffer_size)
while len(data) > 0:
    stream.write(data)
    update_plot(data)
    data = wav_file.readframes(buffer_size)

# Cerrar el flujo y liberar recursos
stream.stop_stream()
stream.close()
p.terminate()
wav_file.close()
"""
