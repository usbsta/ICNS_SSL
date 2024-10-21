import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt
import time  # Import time for sleep

# Constants and configurations
FORMAT = 'int32'  # Data format for numpy arrays
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)  # Duration of chunk in samples
CHUNK_DURATION = CHUNK / RATE  # Duration of chunk in seconds
RECORD_SECONDS = 12000
c = 343
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.5e8

# Paths to your pre-recorded .wav files
wav_filenames2 = [
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_1_nosync.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_2_nosync.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_3_nosync.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/3rd Oct 11/5/device_4_nosync.wav'
]

lowcut = 400.0
highcut = 8000.0

azimuth_range = np.arange(-180, 181, 20)
elevation_range = np.arange(0, 91, 20)

a = [0, -120, -240]
a2 = [-40, -80, -160, -200, -280, -320]

# Microphone positions
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],    # Mic 1  Z3
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],    # Mic 2  Z3
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],    # Mic 3  Z3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],    # Mic 4  Z3
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],    # Mic 5  Z3
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],    # Mic 6  Z3
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],    # Mic 7  Z2
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],    # Mic 8  Z2
    [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],    # Mic 9  Z2
    [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],    # Mic 10 Z2
    [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],    # Mic 11 Z2
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],    # Mic 12 Z2
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],    # Mic 13 Z1
    [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],    # Mic 14 Z1
    [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],    # Mic 15 Z1
    [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],    # Mic 16 Z1
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],    # Mic 17 Z1
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]],    # Mic 18 Z1
    [r[2] * np.cos(np.radians(a2[0])), r[2] * np.sin(np.radians(a2[0])), h[2]],  # Mic 1  Z0
    [r[2] * np.cos(np.radians(a2[1])), r[2] * np.sin(np.radians(a2[1])), h[2]],  # Mic 2  Z0
    [r[2] * np.cos(np.radians(a2[2])), r[2] * np.sin(np.radians(a2[2])), h[2]],  # Mic 3  Z0
    [r[2] * np.cos(np.radians(a2[3])), r[2] * np.sin(np.radians(a2[3])), h[2]],  # Mic 4  Z0
    [r[2] * np.cos(np.radians(a2[4])), r[2] * np.sin(np.radians(a2[4])), h[2]],  # Mic 5  Z0
    [r[2] * np.cos(np.radians(a2[5])), r[2] * np.sin(np.radians(a2[5])), h[2]],  # Mic 6  Z0
])

# Prepare buffers to store audio data
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(4)]  # Synchronized buffers
buffers2 = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(4)]  # Non-synchronized buffers
frames = [[], [], [], []]  # Frames for 4 devices
frames2 = [[], [], [], []]  # Frames for 4 devices (non-synchronized)

# Variables for synchronization
correction_12 = None
correction_13 = None
correction_14 = None

total_correction_12 = 0
total_correction_13 = 0
total_correction_14 = 0

synced_12 = False
synced_13 = False
synced_14 = False

window_size = int(0.2 * RATE)

buffer_lock = threading.Lock()

stop_event = threading.Event()
eof_flags = [False] * 4  # Flags to indicate EOF for each thread

def shift_signal(signal, shift_amount):
    if shift_amount > 0:  # Shift to right (delay)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Shift to left (advance)
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal
    return shifted_signal

def plot_peaks(signals, peaks_indices, labels):
    plt.figure()
    time_axis = np.linspace(0, 200, window_size)
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None and 0 <= peak_idx < len(signal):
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 50 ms Window')
    plt.show()

# Beamforming function
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # 3D direction vector
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            delays = np.dot(mic_positions, direction_vector) / c

            # Applying delays
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1]  # Normalize amplitude with number of mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal

# Modified function to read from .wav files instead of live recording
def record_device_from_wav(wav_filename, buffer_index, stop_event):
    global correction_12, correction_13, correction_14
    global total_correction_12, total_correction_13, total_correction_14
    global eof_flags  # Added this

    try:
        wf = wave.open(wav_filename, 'rb')
        # Check that the parameters match
        assert wf.getnchannels() == CHANNELS
        assert wf.getsampwidth() == 4  # Since we are using int32
        assert wf.getframerate() == RATE

        print(f"Reading from WAV file {wav_filename}...")

        while not stop_event.is_set():
            data = wf.readframes(CHUNK)
            if len(data) == 0:
                # End of file
                eof_flags[buffer_index] = True  # Set EOF flag for this thread
                break

            signal_data = np.frombuffer(data, dtype=np.int32)

            with buffer_lock:
                buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))
                buffers2[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

                # Apply cumulative corrections
                if buffer_index == 1 and total_correction_12 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_13)

                if buffer_index == 3 and total_correction_14 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_14)

                frames[buffer_index].append(buffers[buffer_index].tobytes())
                frames2[buffer_index].append(buffers2[buffer_index].tobytes())

            # Simulate real-time by sleeping for the duration of the chunk
            time.sleep(CHUNK_DURATION)

    except Exception as e:
        print(f"Error reading from WAV file {wav_filename}: {e}")
    finally:
        wf.close()
        print(f"WAV file {wav_filename} closed")

def stop_recording_and_save():
    stop_event.set()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Processing completed.")

    # Optionally, save the processed data to files if needed
    # For simulation purposes, you might skip this part

def signal_handler(sig, frame):
    print("Signal received, stopping processing...")
    stop_recording_and_save()
    sys.exit(0)

# Register signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)

# Start threads to read from WAV files
threads = []
threads.append(threading.Thread(target=record_device_from_wav, args=(wav_filenames2[0], 0, stop_event)))
threads.append(threading.Thread(target=record_device_from_wav, args=(wav_filenames2[1], 1, stop_event)))
threads.append(threading.Thread(target=record_device_from_wav, args=(wav_filenames2[2], 2, stop_event)))
threads.append(threading.Thread(target=record_device_from_wav, args=(wav_filenames2[3], 3, stop_event)))

for thread in threads:
    thread.start()

# Real-time plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_title('Beamforming Energy')

# Max energy point
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

# Main processing loop
while not stop_event.is_set():
    with buffer_lock:
        # Check if buffers have data
        if any(buffer.size == 0 for buffer in buffers):
            # Some buffers are empty, wait for data
            pass  # Do nothing, or you can add a sleep
        else:
            # Proceed with processing
            combined_signal = np.hstack(buffers)
            combined_signal2 = np.hstack(buffers2)

            num_samples = combined_signal.shape[0]

            # Find peaks in the signals
            peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)
            peaks_2, _ = find_peaks(buffers[1][:, 5], height=peak_threshold)
            peaks_3, _ = find_peaks(buffers[2][:, 5], height=peak_threshold)
            peaks_4, _ = find_peaks(buffers[3][:, 5], height=peak_threshold)

            # Plot peaks if synchronization hasn't occurred yet
            if peaks_1.size > 0 and peaks_2.size > 0 and peaks_3.size > 0 and peaks_4.size > 0 and not (synced_12 and synced_13 and synced_14):
                start_idx = max(peaks_1[0] - window_size // 2, 0)
                end_idx = start_idx + window_size

                if end_idx > buffers[0].shape[0]:
                    end_idx = buffers[0].shape[0]
                    start_idx = end_idx - window_size

                signal_1_ch4 = buffers[0][start_idx:end_idx, 5]  # Channel 6 device 1
                signal_2_ch1 = buffers[1][start_idx:end_idx, 5]  # Channel 6 device 2
                signal_3_ch4 = buffers[2][start_idx:end_idx, 5]  # Channel 6 device 3
                signal_4_ch4 = buffers[3][start_idx:end_idx, 5]  # Channel 6 device 4

                signals = [signal_1_ch4, signal_2_ch1, signal_3_ch4, signal_4_ch4]
                peaks_indices = [peaks_1[0] - start_idx, peaks_2[0] - start_idx, peaks_3[0] - start_idx, peaks_4[0] - start_idx]
                labels = ['Device 1 Channel 6', 'Device 2 Channel 6', 'Device 3 Channel 6', 'Device 4 Channel 6']
                plot_peaks(signals, peaks_indices, labels)

            # Synchronization
            if peaks_1.size > 0 and peaks_2.size > 0 and not synced_12:
                peak_time_1 = peaks_1[0] / RATE
                peak_time_2 = peaks_2[0] / RATE
                time_difference_12 = peak_time_2 - peak_time_1
                print(f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds")

                sample_difference_12 = int(time_difference_12 * RATE)

                if abs(time_difference_12) > TARGET_DIFFERENCE:
                    correction_12 = -sample_difference_12
                    total_correction_12 += correction_12
                    buffers[1] = shift_signal(buffers[1], correction_12)
                    print(f"Applying correction of {correction_12} samples to synchronize devices 1 and 2.")
                    synced_12 = True
                    print("Synchronized devices 1 and 2")
                else:
                    print(f"Time difference between devices 1 and 2 is within the target threshold.")

            if peaks_1.size > 0 and peaks_3.size > 0 and not synced_13:
                peak_time_1 = peaks_1[0] / RATE
                peak_time_3 = peaks_3[0] / RATE
                time_difference_13 = peak_time_3 - peak_time_1
                print(f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds")

                sample_difference_13 = int(time_difference_13 * RATE)

                if abs(time_difference_13) > TARGET_DIFFERENCE:
                    correction_13 = -sample_difference_13
                    total_correction_13 += correction_13
                    buffers[2] = shift_signal(buffers[2], correction_13)
                    print(f"Applying correction of {correction_13} samples to synchronize devices 1 and 3.")
                    synced_13 = True
                    print("Synchronized devices 1 and 3")
                else:
                    print(f"Time difference between devices 1 and 3 is within the target threshold.")

            if peaks_1.size > 0 and peaks_4.size > 0 and not synced_14:
                peak_time_1 = peaks_1[0] / RATE
                peak_time_4 = peaks_4[0] / RATE
                time_difference_14 = peak_time_4 - peak_time_1
                print(f"Time difference detected between devices 1 and 4: {time_difference_14:.6f} seconds")

                sample_difference_14 = int(time_difference_14 * RATE)

                if abs(time_difference_14) > TARGET_DIFFERENCE:
                    correction_14 = -sample_difference_14
                    total_correction_14 += correction_14
                    buffers[3] = shift_signal(buffers[3], correction_14)
                    print(f"Applying correction of {correction_14} samples to synchronize devices 1 and 4.")
                    synced_14 = True
                    print("Synchronized devices 1 and 4")
                else:
                    print(f"Time difference between devices 1 and 4 is within the target threshold.")

            # Filtering
            filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

            # Beamforming
            energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

            max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
            estimated_azimuth = azimuth_range[max_energy_idx[0]]
            estimated_elevation = elevation_range[max_energy_idx[1]]

            # Update heatmap
            cax.set_data(energy.T)
            cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Update color limits

            # Update max energy marker
            max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

            # Update text with coordinates
            max_energy_text.set_position((estimated_azimuth, estimated_elevation))
            max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

            fig.canvas.draw()
            fig.canvas.flush_events()

    # After processing, check if all EOF flags are set
    if all(eof_flags):
        stop_event.set()
        print("All WAV files have been processed.")

    # Add a small sleep to prevent tight looping
    #time.sleep(0.01)

# After processing is complete
stop_recording_and_save()
