import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

print(sd.query_devices())
# Configuración de grabación
samplerate = 48000  # Frecuencia de muestreo
duration = 5  # Duración de la grabación en segundos
channels = 12  # Número de canales
device_index = 6  # Índice del dispositivo, cámbialo según lo que encuentres

# Función para grabar audio
def grabar_audio(samplerate, duration, channels, device_index):
    print(f"Iniciando grabación en el dispositivo {device_index}...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32', device=device_index)
    sd.wait()  # Espera a que termine la grabación
    print("Grabación terminada.")
    return audio_data

# Graba el audio
audio_data = grabar_audio(samplerate, duration, channels, device_index)

# Normaliza el audio a rango de -32768 a 32767
audio_data = np.int16(audio_data * 32767)

# Guarda el archivo .wav
output_file = 'grabacion_12_canales.wav'
write(output_file, samplerate, audio_data)
print(f"Archivo guardado como {output_file}")
