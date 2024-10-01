import pyaudio
import wave
import numpy as np

# Configuración de la grabación
FORMAT = pyaudio.paInt16  # Formato de los datos de audio
CHANNELS = 1  # Número de canales
RATE = 44100  # Tasa de muestreo (samples per second)
CHUNK = int(RATE * 0.2)  # Tamaño del bloque de datos para 200 ms (0.2 segundos)
RECORD_SECONDS = 5  # Duración de la grabación en segundos
WAVE_OUTPUT_FILENAME = "output.wav"  # Nombre del archivo de salida

# Inicialización de PyAudio
audio = pyaudio.PyAudio()

# Abrir flujo de entrada
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []

# Bucle para capturar audio y calcular RMS cada 200 ms
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convertir los datos a un array de numpy
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Calcular el RMS
    rms = np.sqrt(np.mean(audio_data ** 2))
    print(f"RMS: {rms}")

print("Grabación completada.")

# Detener y cerrar el flujo
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar los datos grabados en un archivo WAV
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
