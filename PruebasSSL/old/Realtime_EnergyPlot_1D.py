#Beam in time
'''
import numpy as np
import pyaudio
import wave
from scipy.signal import correlate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Audio setup
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # channels or microphones
RATE = 48000  # fs
CHUNK = int(0.2 * RATE)  # buffer size in 200 ms
RECORD_SECONDS = 100  # recording time
#OUTPUT_FILENAME = "output_pyaudio_realtime.wav"
d = 0.04  # distance between mics
c = 343

audio = pyaudio.PyAudio()

device_index = 1 #Win
device_index = 0 #osx

# Create stream object for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("recording...")

frames = []
angles_history = []
energy_history = []

angles_range = np.arange(-90, 91, 1)  # angles in degrees for sweep

# Set up real-time plotting
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # This may need to be adjusted based on the expected energy range
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Energía')

# Capture and real-time processing
for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binaries to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]
    energy = np.zeros(len(angles_range))  # energy in each angle

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        delay_samples = np.round(delays * RATE).astype(int)  # Convert time delays to samples

        # Apply Delay and Sum Beamforming for this angle
        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS):
            delayed_signal = np.roll(signal_data[:, i], -delay_samples[i])
            output_signal += delayed_signal
        output_signal /= CHANNELS  # Normalize

        # Calculate the energy of the combined signal
        energy[idx] = np.sum(output_signal**2)

    # Save the energy for each angle and time
    energy_history.append(energy)
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Ángulo estimado en la ventana actual: {estimated_angle:.2f} grados")

    # Update the real-time plot
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Dynamically adjust the y-axis limit
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Grabación finalizada.")

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

print(f"Archivo de grabación guardado en {OUTPUT_FILENAME}")

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Ángulo promedio de todas las ventanas: {overall_avg_angle:.2f} grados")
else:
    print("No se pudieron calcular ángulos en ninguna ventana.")

'''

 # Beam in frequency





import numpy as np
import pyaudio
import wave
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Audio setup
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # channels or microphones
RATE = 48000  # fs
CHUNK = int(0.2 * RATE)  # buffer size in 200 ms
RECORD_SECONDS = 100  # recording time
# OUTPUT_FILENAME = "output_pyaudio_realtime.wav"
d = 0.04  # distance between mics
c = 343

audio = pyaudio.PyAudio()

device_index = 1  # Win
#device_index = 0  # OSX

# Create stream object for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []
angles_history = []
energy_history = []

angles_range = np.arange(-90, 91, 1)  # angles in degrees for sweep

# Set up real-time plotting
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # This may need to be adjusted based on the expected energy range
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Energy')

# Capture and real-time processing
for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binaries to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]
    energy = np.zeros(len(angles_range))  # energy in each angle

    # FFT of the original signal
    signal_fft = fft(signal_data, axis=0)

    # Frequencies associated
    freqs = np.fft.fftfreq(num_samples, d=1.0/RATE)

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        #delay_phase = np.exp(-1j * 2 * np.pi * freqs[:, None] * delays)  # Calculate the phase term in frequency domain, invert sign than time
        delay_phase = np.exp(1j * 2 * np.pi * freqs[:, None] * delays)  # Calculate the phase term in frequency domain

        # Apply Delay and Sum Beamforming in frequency domain
        output_fft = np.sum(signal_fft * delay_phase, axis=1)

        # Calculate the energy from the combined signal
        output_signal = np.abs(ifft(output_fft))  # Inverse transform to get the signal in time domain
        energy[idx] = np.sum(output_signal**2)

    # Save the energy for each angle and time
    energy_history.append(energy)
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Estimated angle in the current window: {estimated_angle:.2f} degrees")

    # Update the real-time plot
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Dynamically adjust the y-axis limit
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle of all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")


