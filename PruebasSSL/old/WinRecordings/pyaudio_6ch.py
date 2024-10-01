import pyaudio
import wave

# Configuración de parámetros
FORMAT = pyaudio.paInt32  # 24 bits
CHANNELS = 3
RATE = 48000  # Frecuencia de muestreo de 48 kHz
CHUNK = 1024  # Tamaño del buffer
RECORD_SECONDS = 5
OUTPUT_FILENAME = "../simulations/pyroom/output_pyaudio_channel_3.wav"

# Inicializar PyAudio
audio = pyaudio.PyAudio()

# Configurar el dispositivo (asegúrate de usar el índice correcto)
device_index = 6  # Cambia esto al índice de tu Zoom F6

# Crear un stream para grabación
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Grabación finalizada.")

# Detener y cerrar el stream
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar la grabación en un archivo .wav
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Archivo guardado en {OUTPUT_FILENAME}")
