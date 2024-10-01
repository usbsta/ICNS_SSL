import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys

# Configuration parameters
FORMAT = pyaudio.paInt32
CHANNELS = 6
RATE = 48000
d = 0.04  # mic distance
c = 343  # speed of sound
RECORD_SECONDS = 1200
TARGET_DIFFERENCE = 200e-6
peak_threshold = 1e8

device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 2  # Zoom

audio = pyaudio.PyAudio()

angles_range = np.arange(-90, 91, 1)

output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']

# Set up real-time plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)), label='shift')
ax.set_ylim(0, 1)
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Energy')
plt.legend(loc="lower right")

# Window sizes
SYNC_WINDOW_SECONDS = 2  # Sync window in seconds
SUBWINDOW_MS = 200  # Subwindow for calculations in milliseconds
sync_window_size = int(SYNC_WINDOW_SECONDS * RATE)
subwindow_size = int(SUBWINDOW_MS * RATE / 1000)

# Sync buffers and frames storage
sync_buffers = [np.zeros((sync_window_size, CHANNELS), dtype=np.int32) for _ in range(3)]
frames = [[], [], []]  # frames for 3 devices

correction_12 = None
correction_13 = None
total_correction_12 = 0
total_correction_13 = 0
synced_12 = False
synced_13 = False

buffer_lock = threading.Lock()
stop_event = threading.Event()

def shift_signal(signal, shift_amount):
    """Shifts the signal by a certain number of samples."""
    shifted_signal = np.zeros_like(signal)  # Initialize the shifted signal with zeros
    if shift_amount > 0:  # Shift to the right (delaying)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # Shift to the left (advancing)
        shift_amount = abs(shift_amount)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal  # No shift needed
    return shifted_signal


def plot_peaks(signals, peaks_indices, labels):
    """Plot detected peaks within a 200ms subwindow."""
    plt.figure()
    time_axis = np.linspace(0, 200, subwindow_size)
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None:
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 200ms Subwindow')
    plt.show()

def record_device_sync(device_index, buffer_index, stop_event):
    """Records a 2-second window per device for synchronization."""
    global correction_12, correction_13, total_correction_12, total_correction_13
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=sync_window_size)

        print(f"Recording sync window from device {device_index}...")

        while not stop_event.is_set():
            data = stream.read(sync_window_size)
            signal_data = np.frombuffer(data, dtype=np.int32)

            with buffer_lock:
                sync_buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

                # Apply accumulated corrections
                if buffer_index == 1 and total_correction_12 != 0:
                    sync_buffers[buffer_index] = shift_signal(sync_buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    sync_buffers[buffer_index] = shift_signal(sync_buffers[buffer_index], total_correction_13)

                frames[buffer_index].append(sync_buffers[buffer_index].tobytes())

    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print(f"Stream closed for device {device_index}")

def stop_recording_and_save():
    """Stops recording and saves audio files."""
    stop_event.set()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    print("Stopping recording...")

    for i in range(3):
        with wave.open(output_filenames[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames[i]))

    audio.terminate()
    print("Recording saved and audio terminated.")

def signal_handler(sig, frame):
    """Handle the signal to stop the process."""
    print("Signal received, stopping recording...")
    stop_recording_and_save()
    sys.exit(0)

# Handle interrupt signal (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)

# Create recording threads
threads = []
threads.append(threading.Thread(target=record_device_sync, args=(device_index_1, 0, stop_event)))
threads.append(threading.Thread(target=record_device_sync, args=(device_index_2, 1, stop_event)))
threads.append(threading.Thread(target=record_device_sync, args=(device_index_3, 2, stop_event)))

for thread in threads:
    thread.start()

# Main processing loop for synchronization and beamforming
angles_history = []
for _ in range(int(RATE / sync_window_size * RECORD_SECONDS)):
    combined_signal = np.hstack(sync_buffers)
    num_samples = combined_signal.shape[0]

    # Divide the 2-second window into 200ms subwindows
    for subwindow_start in range(0, sync_window_size, subwindow_size):
        subwindow_end = subwindow_start + subwindow_size
        if subwindow_end > sync_window_size:
            break

        subwindow_signal = combined_signal[subwindow_start:subwindow_end, :]
        energy = np.zeros(len(angles_range))

        # Peak detection for synchronization
        peaks_1, _ = find_peaks(sync_buffers[0][:, 0], height=peak_threshold)
        peaks_2, _ = find_peaks(sync_buffers[1][:, 0], height=peak_threshold)
        peaks_3, _ = find_peaks(sync_buffers[2][:, 0], height=peak_threshold)

        if peaks_1.size > 0 and peaks_2.size > 0 and not synced_12:
            peak_time_1 = peaks_1[0] / RATE
            peak_time_2 = peaks_2[0] / RATE
            time_difference_12 = peak_time_2 - peak_time_1
            sample_difference_12 = int(time_difference_12 * RATE)
            if abs(time_difference_12) > TARGET_DIFFERENCE:
                correction_12 = -sample_difference_12
                total_correction_12 += correction_12
                sync_buffers[1] = shift_signal(sync_buffers[1], correction_12)
                print("synchronized 1 and 2")
                synced_12 = True

        if peaks_1.size > 0 and peaks_3.size > 0 and not synced_13:
            peak_time_1 = peaks_1[0] / RATE
            peak_time_3 = peaks_3[0] / RATE
            time_difference_13 = peak_time_3 - peak_time_1
            sample_difference_13 = int(time_difference_13 * RATE)
            if abs(time_difference_13) > TARGET_DIFFERENCE:
                correction_13 = -sample_difference_13
                total_correction_13 += correction_13
                sync_buffers[2] = shift_signal(sync_buffers[2], correction_13)
                print("synchronized 1 and 3")
                synced_13 = True

        # Plot peaks in the subwindow for visualization
        if peaks_1.size > 0 and peaks_2.size > 0 and peaks_3.size > 0:
            signal_1 = sync_buffers[0][subwindow_start:subwindow_end, 0]
            signal_2 = sync_buffers[1][subwindow_start:subwindow_end, 0]
            signal_3 = sync_buffers[2][subwindow_start:subwindow_end, 0]
            signals = [signal_1, signal_2, signal_3]
            peaks_indices = [peaks_1[0], peaks_2[0], peaks_3[0]]
            labels = ['Device 1', 'Device 2', 'Device 3']
            plot_peaks(signals, peaks_indices, labels)

        # Beamforming calculation for 200ms subwindow
        for idx, angle in enumerate(angles_range):
            steering_angle_rad = np.radians(angle)
            delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c
            delay_samples = np.round(delays * RATE).astype(int)

            output_signal = np.zeros(subwindow_size)
            for i in range(CHANNELS * 3):
                delayed_signal = shift_signal(subwindow_signal[:, i], -delay_samples[i])
                output_signal += delayed_signal

            output_signal /= (CHANNELS * 3)
            energy[idx] = np.sum(output_signal ** 2)

        estimated_angle = angles_range[np.argmax(energy)]
        angles_history.append(estimated_angle)

        # Update the plot to show energy over angles in real time
        line.set_ydata(energy)
        ax.set_ylim(0, max(energy) * 1.1)
        fig.canvas.draw()
        fig.canvas.flush_events()

# Stop recording and save audio files
stop_recording_and_save()

# Calculate overall average angle
if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
