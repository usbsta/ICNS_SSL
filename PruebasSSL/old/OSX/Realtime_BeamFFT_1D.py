import numpy as np
import pyaudio
import wave
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Audio setup
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # number of channels (microphones)
RATE = 48000  # sampling frequency
CHUNK = int(0.2 * RATE)  # buffer size for 200 ms
RECORD_SECONDS = 900  # recording time in seconds
d = 0.04  # distance between microphones in meters
c = 343  # speed of sound in m/s

audio = pyaudio.PyAudio()

device_index = 0

# Create a stream object for recording
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

angles_range = np.arange(-90, 91, 2)  # angles in degrees for beamforming

# Set up real-time plotting
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot for energy vs angle
line_energy, = ax1.plot(angles_range, np.zeros(len(angles_range)))
ax1.set_ylim(0, 1)  # Adjust based on expected energy range
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Energy')

# Plot for amplitude vs time (first microphone)
line_amplitude, = ax2.plot(np.zeros(CHUNK))
ax2.set_ylim(-2**25, 2**25 - 1)  # Amplitude for int32 format
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

    num_samples = signal_data.shape[0]  # number of samples per channel

    # Perform FFT on each channel (microphone)
    fft_data = np.fft.fft(signal_data, axis=0)
    freq_bins = np.fft.fftfreq(num_samples, 1 / RATE)

    energy = np.zeros(len(angles_range))  # energy for each angle

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        delay_freqs = np.exp(2j * np.pi * freq_bins[:, np.newaxis] * delays[np.newaxis, :])  # Apply frequency domain delays

        # Apply Delay and Sum Beamforming in the frequency domain
        output_signal_freq = np.sum(fft_data * delay_freqs, axis=1)

        # Convert back to time domain
        output_signal = np.fft.ifft(output_signal_freq)

        # Calculate the energy of the combined signal
        energy[idx] = np.sum(np.abs(output_signal)**2)

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
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Recording saved in {OUTPUT_FILENAME}")

# Create a 3D figure to show energy as a function of angle and time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the X (angles), Y (time), and Z (energy) axes
X, Y = np.meshgrid(angles_range, np.arange(len(energy_history)))
Z = np.array(energy_history)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Axis labels
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Time (window)')
ax.set_zlabel('Energy')

# Show the plot
plt.title('Energy as a function of angle and time')
plt.show()

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
