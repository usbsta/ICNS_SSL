import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

# Cargar señales de audio de los micrófonos
#fs, signals = wavfile.read('simulated_mics2_OSX.wav')
fs, signals = wavfile.read('output_pyaudio_channel_3.wav')
signal1 = signals[:, 0]
signal2 = signals[:, 1]
signal3 = signals[:, 2]

# Calcular la correlación cruzada para TDOA
corr12 = correlate(signal1, signal2, 'full')
corr23 = correlate(signal2, signal3, 'full')

# Encontrar el desplazamiento de mayor correlación (TDOA)
tdoa_12 = np.argmax(corr12) - len(signal1) + 1
tdoa_23 = np.argmax(corr23) - len(signal2) + 1# Convertir TDOA de muestras a tiempo
tdoa_12_time = tdoa_12 / fs
tdoa_23_time = tdoa_23 / fs

# Parámetros
d = 0.04# Distancia entre micrófonos en metros (4 cm)
c = 343# Velocidad del sonido en m/s# Calculando los ángulos para cada par de micrófonos
theta_12 = np.arcsin((tdoa_12_time * c) / d)
theta_23 = np.arcsin((tdoa_23_time * c) / d)

# Promediando los ángulos para estimar la dirección final
theta_avg = np.degrees((theta_12 + theta_23) / 2)

print(f"Ángulo estimado de la fuente sonora: {theta_avg:.2f} grados")
