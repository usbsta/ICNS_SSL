
import pyaudio
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading


# Audio configuration
FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6  # channels per device
RATE = 48000  # sampling rate
CHUNK = int(0.2 * RATE)  # buffer size in 200 ms
d = 0.04  # distance between microphones
c = 343  # speed of sound in m/s
RECORD_SECONDS = 900  # recording time
TARGET_DIFFERENCE = 42e-6  # 42 microseconds, 2 samples time

# Device indices
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

audio = pyaudio.PyAudio()

# Angles to analyze
angles_range = np.arange(-90, 91, 1)

# Prepare the plot for real-time display
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)))
ax.set_ylim(0, 1)  # Adjust according to the expected energy range
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Energy')

# Buffer to store audio data
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

# peak detection
peak_threshold = 1e8

# Variables to store the time corrections
correction_12 = None  # Correction between device 1 and 2
correction_23 = None  # Correction between device 2 and 3

def shift_signal(signal, shift_amount):
    # Shift the signal array to the right or left, filling with zeros
    if shift_amount > 0:  # Shift to the right "Delaying"
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Shift to the left "Advancing"
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal  # No shift applied
    return shifted_signal

def record_device(device_index, buffer_index):
    global correction_12, correction_23

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    print(f"Recording from device {device_index}...")

    while True:
        data = stream.read(CHUNK)
        signal_data = np.frombuffer(data, dtype=np.int32)
        buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

        # Apply consistent time correction on device 2 if detected
        if buffer_index == 1 and correction_12 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_12)

        # Apply consistent time correction on device 3 if detected
        if buffer_index == 2 and correction_23 is not None:
            buffers[buffer_index] = shift_signal(buffers[buffer_index], correction_23)

    stream.stop_stream()
    stream.close()
    '''
    with wav.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes('b'.join(frames))
    '''

# Start recording threads
thread_1 = threading.Thread(target=record_device, args=(device_index_1, 0))
thread_2 = threading.Thread(target=record_device, args=(device_index_2, 1))
thread_3 = threading.Thread(target=record_device, args=(device_index_3, 2))

thread_1.start()
thread_2.start()
thread_3.start()

# Real-time processing
angles_history = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    # Combine the data from the three devices
    combined_signal = np.hstack(buffers)
    num_samples = combined_signal.shape[0]
    energy = np.zeros(len(angles_range))  # energy at each angle

    # Peak detection in the channels of interest
    peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)  # Channel 6 of device 1
    peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)  # Channel 1 of device 2
    peaks_3, _ = find_peaks(buffers[1][:, 5], height=peak_threshold)  # Channel 6 of device 2
    peaks_4, _ = find_peaks(buffers[2][:, 0], height=peak_threshold)  # Channel 1 of device 3

    # Calculate the time difference between the detected peaks for devices 1 and 2
    if peaks_1.size > 0 and peaks_2.size > 0:
        peak_time_1 = peaks_1[0] / RATE  # Time of the first peak on channel 6 of device 1
        peak_time_2 = peaks_2[0] / RATE  # Time of the first peak on channel 1 of device 2
        time_difference_12 = peak_time_2 - peak_time_1
        print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds")

        sample_difference_12 = int(time_difference_12 * RATE)

        # Apply correction if the difference is greater than 42 microseconds
        if abs(time_difference_12) > TARGET_DIFFERENCE:
            correction_12 = -sample_difference_12
            buffers[1] = shift_signal(buffers[1], correction_12)
            print(f"Applying correction of {correction_12} samples to synchronize devices 1 and 2.")
        else:
            print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds (less than 42 us, no further adjustment)")

    # Calculate the time difference between the detected peaks for devices 2 and 3
    if peaks_3.size > 0 and peaks_4.size > 0:
        peak_time_3 = peaks_3[0] / RATE  # Time of the first peak on channel 6 of device 2
        peak_time_4 = peaks_4[0] / RATE  # Time of the first peak on channel 1 of device 3
        time_difference_23 = peak_time_4 - peak_time_3
        print(f"Time difference detected between devices 2 and 3: {time_difference_23:.6f} seconds")

        sample_difference_23 = int(time_difference_23 * RATE)

        # Apply correction if the difference is greater than 42 microseconds
        if abs(time_difference_23) > TARGET_DIFFERENCE:
            correction_23 = -sample_difference_23
            buffers[2] = shift_signal(buffers[2], correction_23)
            print(f"Applying correction of {correction_23} samples to synchronize devices 2 and 3.")
        else:
            print(f"Time difference detected between devices 2 and 3: {time_difference_23:.6f} seconds (less than 42 us, no further adjustment)")

    # Iterate over all possible angles to find the direction with maximum energy
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)  # Convert angle to radians
        delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c  # Calculate time delays
        delay_samples = np.round(delays * RATE).astype(int)  # Convert delays to samples

        # Apply Delay-and-Sum Beamforming for this angle
        output_signal = np.zeros(num_samples)
        for i in range(CHANNELS * 3):
            delayed_signal = shift_signal(combined_signal[:, i], -delay_samples[i])
            output_signal += delayed_signal
        output_signal /= (CHANNELS * 3)  # Normalize

        # Calculate the energy of the combined signal
        energy[idx] = np.sum(output_signal ** 2)

    # Save the energy for each angle and time
    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)
    print(f"Estimated angle in the current window: {estimated_angle:.2f} degrees")

    # Update the plot in real time
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Dynamically adjust the y-axis limit
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Beamforming processing completed.")

# Terminate the audio session
audio.terminate()

# If you want to average all calculated angles
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
