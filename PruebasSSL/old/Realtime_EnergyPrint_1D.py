#Processing BEAMFORMING IN TIME
'''
import numpy as np
import pyaudio
import wave
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Configuración de parámetros
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 3  # Número de canales (micrófonos)
RATE = 48000  # Frecuencia de muestreo (48 kHz)
CHUNK = int(0.2 * RATE)  # Tamaño del buffer para 200 ms
RECORD_SECONDS = 10  # Tiempo total de grabación
#OUTPUT_FILENAME = "output_pyaudio_realtime.wav"  # Parámetros para la estimación del ángulo
d = 0.04  # Distancia entre micrófonos en metros (4 cm)
c = 343  # Velocidad del sonido en m/s

# Inicializar PyAudio
audio = pyaudio.PyAudio()

# Configurar el dispositivo (asegúrate de usar el índice correcto)
device_index = 6  # Win
device_index = 0  # OSX

# Crear un stream para grabación
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Grabando en tiempo real...")

frames = []
angles = []
angles_history = []

# Rango de ángulos a evaluar (de -90 a 90 grados)
angles_range = np.arange(-90, 91, 1)  # Ángulos en grados

# Captura y procesamiento en tiempo real
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convertir los datos binarios en un array numpy
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]
    energy = np.zeros(len(angles_range))  # Energía en cada ángulo

    # Iterar sobre todos los ángulos posibles para encontrar la dirección con máxima energía
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convertir ángulo a radianes
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calcular retrasos en tiempo
        delay_samples = np.round(delays * RATE).astype(int)  # Convertir retrasos de tiempo a muestras

        # Aplicar Delay and Sum Beamforming para este ángulo
        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS):
            delayed_signal = np.roll(signal_data[:, i], -delay_samples[i])
            output_signal += delayed_signal
        output_signal /= CHANNELS  # Normalizar

        # Calcular la energía de la señal combinada
        energy[idx] = np.sum(output_signal**2)

    # Encontrar el ángulo con la máxima energía
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Ángulo estimado en la ventana actual: {estimated_angle:.2f} grados")

print("Grabación finalizada.")

# Detener y cerrar el stream
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar la grabación completa en un archivo .wav
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Archivo de grabación guardado en {OUTPUT_FILENAME}")

# Si quieres promediar todos los ángulos calculados
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Ángulo promedio de todas las ventanas: {overall_avg_angle:.2f} grados")
else:
    print("No se pudieron calcular ángulos en ninguna ventana.")

# Graficar la energía en función del ángulo en la última ventana
plt.plot(angles_range, energy)
plt.title('Energía del haz en función del ángulo (última ventana)')
plt.xlabel('Ángulo (grados)')
plt.ylabel('Energía')
plt.show()
'''





# processing BEAMFORMING in frequency



import numpy as np
import pyaudio
import wave
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Parameter configuration
FORMAT = pyaudio.paInt32  # 32-bit
CHANNELS = 6  # Number of channels (microphones)
RATE = 48000  # Sampling rate (48 kHz)
CHUNK = int(0.2 * RATE)  # Buffer size for 200 ms
RECORD_SECONDS = 10  # Total recording time
# OUTPUT_FILENAME = "output_pyaudio_realtime_fft.wav"  # File name for saving the recording
d = 0.04  # Distance between microphones in meters (4 cm)
c = 343  # Speed of sound in m/s

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Configure the device (make sure to use the correct index)
device_index = 6  # Win
device_index = 0  # OSX

# Create a stream for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Recording in real-time...")

frames = []
angles_history = []

# Range of angles to evaluate (from -90 to 90 degrees)
angles_range = np.arange(-90, 91, 1)  # Angles in degrees

# Real-time capture and processing
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binary data into a numpy array
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]
    energy = np.zeros(len(angles_range))  # Energy at each angle

    # FFT of the original signal
    signal_fft = fft(signal_data, axis=0)

    # Associated frequencies
    freqs = np.fft.fftfreq(num_samples, d=1.0/RATE)

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        #delay_phase = np.exp(-1j * 2 * np.pi * freqs[:, None] * delays)  # Calculate the phase term in frequency domain, invert sign than time
        delay_phase = np.exp(1j * 2 * np.pi * freqs[:, None] * delays)  # direct relation in sign with time calculations
        # Apply Delay and Sum Beamforming in frequency domain
        output_fft = np.sum(signal_fft * delay_phase, axis=1)

        # Calculate the energy from the combined signal
        output_signal = np.abs(ifft(output_fft))  # Inverse transform to get the signal in time domain
        energy[idx] = np.sum(output_signal**2)

    # Find the angle with the maximum energy
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Estimated angle in the current window: {estimated_angle:.2f} degrees")

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the complete recording in a .wav file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Recording file saved as {OUTPUT_FILENAME}")

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle of all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")

# Plot the energy as a function of angle in the last window
plt.plot(angles_range, energy)
plt.title('Beam energy as a function of angle (last window) - FFT')
plt.xlabel('Angle (degrees)')
plt.ylabel('Energy')
plt.show()

