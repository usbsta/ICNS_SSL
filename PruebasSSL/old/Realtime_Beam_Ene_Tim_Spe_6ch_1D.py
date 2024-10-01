# Beam in time

'''
import numpy as np
import pyaudio
import wave
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Audio setup
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # channels or microphones
RATE = 48000  # fs
CHUNK = int(0.2 * RATE)  # buffer size in 200 ms
RECORD_SECONDS = 900  # recording time
#OUTPUT_FILENAME = "output_pyaudio_realtime6.wav"
d = 0.04  # distance between mics
c = 343

audio = pyaudio.PyAudio()

device_index = 0

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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot for energy vs angle
line_energy, = ax1.plot(angles_range, np.zeros(len(angles_range)))
ax1.set_ylim(0, 1)  # Adjust based on expected energy range
ax1.set_xlabel('Angle (degree)')
ax1.set_ylabel('Energy')

# Plot for amplitude vs time (first microphone)
line_amplitude, = ax2.plot(np.zeros(CHUNK))
#ax2.set_ylim(-2**31, 2**31 - 1)  # Amplitud para formato int32
ax2.set_ylim(-2**25, 2**25 - 1)  # Amplitud para formato int32
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitud')

# Spectrogram plot
spectrum, freqs, bins, im = ax3.specgram(np.zeros(CHUNK), NFFT=1024, Fs=RATE, noverlap=512, cmap='viridis')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Frequency (Hz)')

# Capture and real-time processing
for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binaries to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0] # number of signals or microphones
    energy = np.zeros(len(angles_range))  # energy in each angle

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        delay_samples = np.round(delays * RATE).astype(int)  # Convert time delays to samples

        # Apply Delay and Sum Beamforming for this angle
        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS):
            delayed_signal = np.roll(signal_data[:, i], -delay_samples[i]) # shift or move the signal
            output_signal += delayed_signal # summing
        output_signal /= CHANNELS  # Normalize

        # Calculate the energy of the combined signal
        energy[idx] = np.sum(output_signal**2)

    # Save the energy for each angle and time
    energy_history.append(energy)
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Ángulo estimado en la ventana actual: {estimated_angle:.2f} grados")

    # Update the real-time plot for energy vs angle
    line_energy.set_ydata(energy)
    ax1.set_ylim(0, max(energy) * 1.1)  # Dynamically adjust the y-axis limit

    # Update the real-time plot for amplitude vs time (first microphone)
    line_amplitude.set_ydata(signal_data[:, 0])

    # Update the real-time spectrogram
    ax3.cla()
    ax3.specgram(signal_data[:, 0], NFFT=1024, Fs=RATE, noverlap=512, cmap='viridis')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Frecuencia (Hz)')

    # Draw and update all plots
    plt.pause(0.01)

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

# Create a 3D figure to show energy as a function of angle and time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the X (angles), Y (time), and Z (energy) axes
X, Y = np.meshgrid(angles_range, np.arange(len(energy_history)))
Z = np.array(energy_history)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Axis labels
ax.set_xlabel('Ángulo (grados)')
ax.set_ylabel('Tiempo (ventana)')
ax.set_zlabel('Energía')

# Show the plot
plt.title('Energía en función del ángulo y del tiempo')
plt.show()

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Ángulo promedio de todas las ventanas: {overall_avg_angle:.2f} grados")
else:
    print("No se pudieron calcular ángulos en ninguna ventana.")
'''

# Beam in Freq


import numpy as np
import pyaudio
import wave
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Audio setup
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # number of microphones
RATE = 48000  # sampling rate in Hz
CHUNK = int(0.3 * RATE)  # buffer size in 300 ms
RECORD_SECONDS = 900  # total recording time in seconds
# OUTPUT_FILENAME = "output_pyaudio_realtime6.wav"
d = 0.04  # distance between microphones in meters
c = 343  # speed of sound in m/s

audio = pyaudio.PyAudio()

device_index = 0

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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot for energy vs angle
line_energy, = ax1.plot(angles_range, np.zeros(len(angles_range)))
ax1.set_ylim(0, 1)  # Adjust based on expected energy range
ax1.set_xlabel('Angle (degree)')
ax1.set_ylabel('Energy')

# Plot for amplitude vs time (first microphone)
line_amplitude, = ax2.plot(np.zeros(CHUNK))
ax2.set_ylim(-2**25, 2**25 - 1)  # Amplitude range for int32 format
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')

# Spectrogram plot
spectrum, freqs, bins, im = ax3.specgram(np.zeros(CHUNK), NFFT=1024, Fs=RATE, noverlap=512, cmap='viridis')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Frequency (Hz)')

# Capture and real-time processing
for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # Convert binary data to numpy arrays
    signal_data = np.frombuffer(data, dtype=np.int32)
    signal_data = np.reshape(signal_data, (-1, CHANNELS))

    num_samples = signal_data.shape[0]  # number of samples
    energy = np.zeros(len(angles_range))  # energy for each angle

    # FFT of the original signal
    signal_fft = fft(signal_data, axis=0)

    # Frequencies associated
    freqs = np.fft.fftfreq(num_samples, d=1.0 / RATE)

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        delay_phase = np.exp(1j * 2 * np.pi * freqs[:, None] * delays)  # Calculate phase shifts for delays

        # Apply Delay and Sum Beamforming in the frequency domain
        output_fft = np.sum(signal_fft * delay_phase, axis=1)

        # Convert back to the time domain
        output_signal = np.abs(ifft(output_fft))

        # Calculate the energy of the combined signal
        energy[idx] = np.sum(output_signal**2)

    # Save the energy for each angle and time
    energy_history.append(energy)
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Estimated angle in the current window: {estimated_angle:.2f} degrees")

    # Update the real-time plot for energy vs angle
    line_energy.set_ydata(energy)
    ax1.set_ylim(0, max(energy) * 1.1)  # Dynamically adjust the y-axis limit

    # Update the real-time plot for amplitude vs time (first microphone)
    line_amplitude.set_ydata(signal_data[:, 0])

    # Update the real-time spectrogram
    ax3.cla()
    ax3.specgram(signal_data[:, 0], NFFT=1024, Fs=RATE, noverlap=512, cmap='viridis')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')

    # Draw and update all plots
    plt.pause(0.01)

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the complete recording in a .wav file
# with wave.open(OUTPUT_FILENAME, 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(audio.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))

# print(f"Recording file saved as {OUTPUT_FILENAME}")


# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle of all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")


