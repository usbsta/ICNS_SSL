import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Cargar señales de audio de los micrófonos
#fs, signals = wavfile.read('simulated_mics2_OSX.wav')
fs, signals = wavfile.read('output_pyaudio_channel_3.wav')
num_mics = signals.shape[1]  # Número de micrófonos
num_samples = signals.shape[0]  # Número de muestras por señal
# Parámetros físicos
d = 0.04# Distancia entre micrófonos en metros (4 cm)
c = 343# Velocidad del sonido en m/s
# Rango de ángulos a evaluar (de -90 a 90 grados)
angles = np.arange(-90, 91, 1)  # Ángulos en grados
energy = np.zeros(len(angles))  # Energía en cada ángulo
# Iterar sobre todos los ángulos posibles para encontrar la dirección con máxima energía
for idx, angle in enumerate(angles):
    steering_angle_rad = np.radians(angle)  # Convertir ángulo a radianes
    delays = np.arange(num_mics) * d * np.sin(steering_angle_rad) / c  # Calcular retrasos en tiempo
    delay_samples = np.round(delays * fs).astype(int)  # Convertir retrasos de tiempo a muestras
    # Aplicar Delay and Sum Beamforming para este ángulo
    output_signal = np.zeros(num_samples)
    for i in range(num_mics):
        delayed_signal = np.roll(signals[:, i], -delay_samples[i])
        output_signal += delayed_signal
    output_signal /= num_mics  # Normalizar
    # Calcular la energía de la señal combinada
    energy[idx] = np.sum(output_signal**2)

# Encontrar el ángulo con la máxima energía
estimated_angle = angles[np.argmax(energy)]

print(f"Ángulo estimado de la fuente sonora: {estimated_angle} grados")

# Graficar la energía en función del ángulo
plt.plot(angles, energy)
plt.title('Energía del haz en función del ángulo')
plt.xlabel('Ángulo (grados)')
plt.ylabel('Energía')
plt.show()
