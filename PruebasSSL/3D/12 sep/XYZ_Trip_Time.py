import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt

# Parámetros
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # número de canales o micrófonos
RATE = 48000  # tasa de muestreo
CHUNK = int(0.2 * RATE)  # tamaño de buffer para 100 ms
RECORD_SECONDS = 900  # tiempo de grabación
d = 0.04  # distancia entre micrófonos
c = 343  # velocidad del sonido en m/s
device_index = 1

# Posiciones de los micrófonos en 3D
mic_positions = np.array([
    [0.12 * np.cos(np.radians(0)), 0.12 * np.sin(np.radians(0)), -0.25],  # Mic 1
    [0.12 * np.cos(np.radians(120)), 0.12 * np.sin(np.radians(120)), -0.25],  # Mic 2
    [0.12 * np.cos(np.radians(240)), 0.12 * np.sin(np.radians(240)), -0.25],  # Mic 3
    [0.2 * np.cos(np.radians(0)), 0.2 * np.sin(np.radians(0)), -0.5],  # Mic 4
    [0.2 * np.cos(np.radians(120)), 0.2 * np.sin(np.radians(120)), -0.5],  # Mic 5
    [0.2 * np.cos(np.radians(240)), 0.2 * np.sin(np.radians(240)), -0.5]  # Mic 6
])

def np_shift(arr, num_shift):
    if num_shift > 0:
        return np.concatenate([np.zeros(num_shift), arr[:-num_shift]])
    elif num_shift < 0:
        return np.concatenate([arr[-num_shift:], np.zeros(-num_shift)])
    else:
        return arr

# Función de beamforming en el dominio del tiempo
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # Vector de dirección en 3D
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            # Cálculo de los retrasos en tiempo para cada micrófono
            delays = np.dot(mic_positions, direction_vector) / c

            # Aplicar los retrasos alineando las señales
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np_shift(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1]  # Normalizar por el número de micrófonos
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)

            '''    
                sample_shift = int(np.round(delay * RATE))  # Convertir el retraso en muestras
                if sample_shift > 0:
                    output_signal[:num_samples-sample_shift] += signal_data[sample_shift:, i]
                else:
                    output_signal[-sample_shift:] += signal_data[:num_samples+sample_shift, i]

            # Cálculo de la energía total
            energy[az_idx, el_idx] = np.sum(output_signal**2)
            '''

    return energy

# Configuración de audio
audio = pyaudio.PyAudio()

# Crear el objeto de stream para grabar
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando...")

frames = []
azimuth_range = np.arange(-90, 91, 10)  # Ángulos de azimut de -90° a 90°
azimuth_range = np.arange(-180, 180, 5)  # Ángulos de azimut de -90° a 90°
elevation_range = np.arange(-90, 91, 5)  # Ángulos de elevación de -90° a 90°

# Configuración de la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energía')
ax.set_xlabel('Azimut (grados)')
ax.set_ylabel('Elevación (grados)')
ax.set_title('Energía del beamforming')

for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convertir los datos binarios a arreglos numpy
    signal_data = np.frombuffer(data, dtype=np.int32).reshape(-1, CHANNELS)

    # Calcular la energía utilizando beamforming en el dominio del tiempo
    energy = beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

    max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
    estimated_azimuth = azimuth_range[max_energy_idx[0]]
    estimated_elevation = elevation_range[max_energy_idx[1]]
    print(f"Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

    # Actualizar los datos del mapa de calor
    cax.set_data(energy.T)
    cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Grabación terminada.")

# Detener y cerrar el stream
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar la grabación completa en un archivo .wav
with wave.open("output_pyaudio_realtime3D_6ch_tiempo.wav", 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Grabación guardada en output_pyaudio_realtime3D_6ch_tiempo.wav")
