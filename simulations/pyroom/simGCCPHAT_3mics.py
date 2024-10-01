import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    Calcula el GCC-PHAT entre dos señales.

    Args:
        sig (ndarray): Señal recibida en el primer micrófono.
        refsig (ndarray): Señal recibida en el segundo micrófono.
        fs (int): Frecuencia de muestreo.
        max_tau (float): Máximo retardo esperado en segundos.
        interp (int): Factor de interpolación para mejorar la resolución.

    Returns:
        (float): Retardo de tiempo estimado entre las señales.
    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift

    return shift / float(interp * fs)


# Cargar señales de audio de los micrófonos
fs, signals = wavfile.read('simulated_mics2_OSX.wav')
num_mics = signals.shape[1]  # Número de micrófonos
num_samples = signals.shape[0]  # Número de muestras por señal
#

# Parámetros físicos
d = 0.04  # Distancia entre micrófonos en metros (4 cm)
c = 343  # Velocidad del sonido en m/s

# Rango de ángulos a evaluar (de -90 a 90 grados)
angles = np.arange(-90, 91, 1)  # Ángulos en grados
energy = np.zeros(len(angles))  # Energía en cada ángulo

# Iterar sobre todos los ángulos posibles para encontrar la dirección con máxima energía
for idx, angle in enumerate(angles):
    steering_angle_rad = np.radians(angle)  # Convertir ángulo a radianes
    delays = np.arange(num_mics) * d * np.sin(steering_angle_rad) / c  # Calcular retrasos en tiempo
    delay_samples = np.round(delays * fs).astype(int)  # Convertir retrasos de tiempo a muestras

    # Aplicar GCC-PHAT para obtener el TDOA
    output_signal = np.zeros(num_samples)
    for i in range(1, num_mics):
        tdoa = gcc_phat(signals[:, 0], signals[:, i], fs=fs)
        # Ajustar la señal de cada micrófono según el TDOA
        delay_samples[i] = int(np.round(tdoa * fs))
        delayed_signal = np.roll(signals[:, i], -delay_samples[i])
        output_signal += delayed_signal

    output_signal /= num_mics  # Normalizar

    # Calcular la energía de la señal combinada
    energy[idx] = np.sum(output_signal ** 2)

# Encontrar el ángulo con la máxima energía
estimated_angle = angles[np.argmax(energy)]

print(f"Ángulo estimado de la fuente sonora: {estimated_angle} grados")

# Graficar la energía en función del ángulo
plt.plot(angles, energy)
plt.title('Energía del haz en función del ángulo (GCC-PHAT)')
plt.xlabel('Ángulo (grados)')
plt.ylabel('Energía')
plt.show()
