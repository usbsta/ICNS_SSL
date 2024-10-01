import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
from scipy.signal import resample

# Parámetros de la sala y el arreglo
room_dim = [4, 4, 3]  # Dimensiones de la sala: 5x5x3 metros
mic_distance = 0.04 # Distancia entre micrófonos: 4 cm
# Crear la sala
fs_simulation = 48000# Frecuencia de muestreo para la simulación
absorption_coefficients = 1.0# Coeficiente de absorción de 1 para todas las superficies
room = pra.ShoeBox(room_dim, fs=fs_simulation, max_order=1, materials=pra.Material(absorption_coefficients))

# Posición de la fuente sonora a 0 grados de la fuente
source_position = [1, 3, 1.5]


# Posiciones del arreglo de micrófonos
mic_center = [1, 2, 1.5]  # Centro del arreglo

# Posición de la fuente sonora a 45 grados
#source_position = [mic_center[0] + np.cos(np.radians(45)),
#                   mic_center[1] + np.sin(np.radians(45)),
#                   mic_center[2]]  # Manteniendo la misma altura

print(source_position)

mic_positions = np.array([
    mic_center,
    [mic_center[0] + mic_distance, mic_center[1], mic_center[2]],
    [mic_center[0] + 2 * mic_distance, mic_center[1], mic_center[2]]
]).T

# Cargar el archivo de audio
audio_signal, fs_original = sf.read("voice.wav")

# Asegurarse de que la señal sea mono
if audio_signal.ndim > 1:
    audio_signal = np.mean(audio_signal, axis=1)

# Resamplear si la frecuencia de muestreo original es diferente a la de la simulaciónif fs_original != fs_simulation:
    num_samples = int(len(audio_signal) * fs_simulation / fs_original)
    audio_signal = resample(audio_signal, num_samples)

# Agregar la fuente y el arreglo de micrófonos a la sala
room.add_source(source_position, signal=audio_signal)
room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

# Simular la propagación del sonido
room.simulate()

# Guardar los audios simulados en un archivo .wav de 3 canales
output_wav = np.array(room.mic_array.signals, dtype=np.float32).T
wavfile.write("simulated_mics2_OSX.wav", room.fs, output_wav)

# Mostrar la configuración de la sala
room.plot(img_order=0)
plt.title("Simulación de la sala con fuente sonora y arreglo de micrófonos")
plt.show()
