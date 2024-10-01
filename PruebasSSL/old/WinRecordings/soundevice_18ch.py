import sounddevice as sd
import numpy as np
import soundfile as sf

# Configuración de dispositivos y grabación
device_indices = [1, 2, 3]  # Reemplaza con los índices de tus grabadoras
channels_per_device = 6  # Número de canales por grabadora
total_channels = channels_per_device * len(device_indices)
fs = 44100  # Frecuencia de muestreo
output_filename = 'output_multichannel.wav'

# Buffer para almacenar los datos grabados
recorded_data = []


def audio_callback(indata, frames, time, status, index):
    if status:
        print(f"Estado del dispositivo {index}: {status}")

    # Agrega los datos de esta grabadora a la posición correspondiente en recorded_data
    if index < len(recorded_data):
        recorded_data[index].append(indata.copy())
    else:
        recorded_data.append([indata.copy()])


# Crear streams de entrada para cada grabadora
streams = []
for i, device_index in enumerate(device_indices):
    stream = sd.InputStream(device=device_index,
                            channels=channels_per_device,
                            samplerate=fs,
                            callback=lambda indata, frames, time, status, idx=i: audio_callback(indata, frames, time,
                                                                                                status, idx))
    streams.append(stream)

# Abrir todos los streams
for stream in streams:
    stream.start()

print(f"Capturando y grabando {total_channels} canales en total. Presiona Ctrl+C para detener.")

try:
    # Mantener el script en ejecución para capturar audio
    while True:
        sd.sleep(1000)
except KeyboardInterrupt:
    print("Finalizando captura de audio y guardando el archivo...")
finally:
    # Asegúrate de cerrar todos los streams
    for stream in streams:
        stream.stop()
        stream.close()

    # Verificar que se hayan grabado datos
    if recorded_data and len(recorded_data) == len(device_indices):
        # Combina los datos de todas las grabadoras en un solo array
        combined_data = np.hstack([np.vstack(device_data) for device_data in recorded_data])

        # Guardar los datos combinados en un archivo WAV
        sf.write(output_filename, combined_data, fs)
        print(f"Grabación guardada en '{output_filename}'")
    else:
        print("No se grabaron datos.")
