
import pyaudio
import wave
import numpy as np

#p = pyaudio.PyAudio()
#for i in range(p.get_device_count()):
#    print (p.get_device_info_by_index(i))

# Configuración de grabación
samplerate = 48000  # Frecuencia de muestreo
duration = 5  # Duración de la grabación en segundos
channels = 12  # Número de canales
device_index = 6  # Índice del dispositivo (ajústalo según tu dispositivo)
CHUNK = (1/samplerate)

# Configuración de PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=channels,
                rate=samplerate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024)

print("Iniciando grabación...")

frames = []

for _ in range(0, int(samplerate / 1024 * duration)):
    data = stream.read(1024)
    frames.append(data)

print("Grabación terminada.")

# Detener y cerrar el stream
stream.stop_stream()
stream.close()
p.terminate()

# Guardar la grabación en un archivo WAV
output_file = 'grabacion_12_canales.wav'
wf = wave.open(output_file, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(samplerate)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Archivo guardado como {output_file}")
