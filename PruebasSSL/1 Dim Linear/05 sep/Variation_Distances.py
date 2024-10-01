import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys

FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
d = 0.1  # mic distance
d = 0.04  # mic distance
distances = np.array([0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28])
#inverted_distances = distances[::-1]

#print(inverted_distances)


c = 343
RECORD_SECONDS = 1200000
TARGET_DIFFERENCE = 200e-6
peak_threshold = 1e8

#without umc204
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 1  # Zoom

# usb c
device_index_1 = 3  # Zoom 3
device_index_2 = 4  # Zoom 2
device_index_3 = 5  # Zoom



audio = pyaudio.PyAudio()

angles_range = np.arange(-90, 91, 1)

output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']
output_filenames2 = ['device_1_nosync.wav', 'device_2_nosync.wav', 'device_3_nosync.wav']

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(angles_range, np.zeros(len(angles_range)),label='shift')
line2, = ax.plot(angles_range, np.zeros(len(angles_range)),label='roll')
line3, = ax.plot(angles_range, np.zeros(len(angles_range)),label='no sync')
ax.set_ylim(0, 1)  # Ajustar según rango de energía esperado
ax.set_xlabel('Angle (degres)')
ax.set_ylabel('Energy')
plt.legend(loc="lower right")

# buffers preparation to store audio
buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)] # syncro
buffers2 = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]
frames = [[], [], []]  # frames for 3 devices
frames2 = [[], [], []]  # frames for 3 devices

correction_12 = None  # correction between 1 and 2
correction_13 = None

total_correction_12 = 0
total_correction_13 = 0

synced_12 = False
synced_13 = False

window_size = int(0.05 * RATE)

buffer_lock = threading.Lock()

stop_event = threading.Event()


def shift_signal(signal, shift_amount):
    if shift_amount > 0:  # shift to right "Retrasando"
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:  # shift to left "Avanzando"
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal
    return shifted_signal


def plot_peaks(signals, peaks_indices, labels):
    plt.figure()
    time_axis = np.linspace(0, 50, window_size)
    for signal, peak_idx, label in zip(signals, peaks_indices, labels):
        plt.plot(time_axis, signal, label=label)
        if peak_idx is not None:
            plt.plot(time_axis[peak_idx], signal[peak_idx], 'ro')
            plt.text(time_axis[peak_idx], signal[peak_idx], f'Peak {peak_idx}', color='red')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Detected Peaks in 50 ms Window')
    plt.show()


def record_device(device_index, buffer_index, stop_event):
    global correction_12, correction_13, total_correction_12, total_correction_13

    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)

        print(f"Recording from device {device_index}...")

        while not stop_event.is_set():
            data = stream.read(CHUNK)
            signal_data = np.frombuffer(data, dtype=np.int32)

            with buffer_lock:
                buffers[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))
                buffers2[buffer_index] = np.reshape(signal_data, (-1, CHANNELS))

                # Aplica correcciones acumuladas
                if buffer_index == 1 and total_correction_12 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_13)

                frames[buffer_index].append(buffers[buffer_index].tobytes())
                frames2[buffer_index].append(buffers2[buffer_index].tobytes())

    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print(f"Stream closed for device {device_index}")


def stop_recording_and_save():
    stop_event.set()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    print("Stopping recording...")

    for i in range(3):
        with wave.open(output_filenames[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames[i]))

        with wave.open(output_filenames2[i], 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames2[i]))

    audio.terminate()
    print("Recording saved and audio terminated.")


def signal_handler(sig, frame):
    print("Signal received, stopping recording...")
    stop_recording_and_save()
    sys.exit(0)


# (Ctrl + C) stop process
signal.signal(signal.SIGINT, signal_handler)

threads = []
threads.append(threading.Thread(target=record_device, args=(device_index_1, 0, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_2, 1, stop_event)))
threads.append(threading.Thread(target=record_device, args=(device_index_3, 2, stop_event)))

for thread in threads:
    thread.start()

# real time processing
angles_history = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    combined_signal = np.hstack(buffers)
    combined_signal2 = np.hstack(buffers2)
    num_samples = combined_signal.shape[0]
    energy = np.zeros(len(angles_range)) # for shift
    energy2 = np.zeros(len(angles_range)) # for roll
    energy3 = np.zeros(len(angles_range)) # without sync


    # peack detection
    peaks_1, _ = find_peaks(buffers[0][:, 5], height=peak_threshold)
    peaks_2, _ = find_peaks(buffers[1][:, 0], height=peak_threshold)
    peaks_4, _ = find_peaks(buffers[2][:, 5], height=peak_threshold)

    if peaks_1.size > 0 and peaks_2.size > 0 and peaks_4.size > 0:
        # Extraer una ventana de 3 ms alrededor de cada pico para graficar
        start_idx = max(peaks_1[0] - window_size // 2, 0)
        end_idx = start_idx + window_size

        # Asegurarse de que end_idx no exceda la longitud del buffer
        if end_idx > buffers[0].shape[0]:
            end_idx = buffers[0].shape[0]
            start_idx = end_idx - window_size

        signal_1_ch5 = buffers[0][start_idx:end_idx, 5]
        signal_2_ch1 = buffers[1][start_idx:end_idx, 0]
        signal_3_ch1 = buffers[2][start_idx:end_idx, 5]

        signals = [signal_1_ch5, signal_2_ch1, signal_3_ch1]
        peaks_indices = [peaks_1[0] - start_idx, peaks_2[0] - start_idx, peaks_4[0] - start_idx]
        labels = ['Device 1 Channel 5', 'Device 2 Channel 1', 'Device 3 Channel 1']
        plot_peaks(signals, peaks_indices, labels)

    # synchronization
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
            synced_12 = True  # flag after synch 1 and 2
            print("synchronized 1 and 2")

        else:
            print(
                f"Time difference detected between devices 1 and 2: {time_difference_12:.6f} seconds (less than TARGET_DIFFERENCE, no further adjustment)")

    if peaks_1.size > 0 and peaks_4.size > 0 and not synced_13:
        peak_time_1 = peaks_1[0] / RATE
        peak_time_4 = peaks_4[0] / RATE
        time_difference_13 = peak_time_4 - peak_time_1
        print(f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds")

        sample_difference_13 = int(time_difference_13 * RATE)

        if abs(time_difference_13) > TARGET_DIFFERENCE:
            correction_13 = -sample_difference_13
            total_correction_13 += correction_13
            buffers[2] = shift_signal(buffers[2], correction_13)
            print(f"Applying correction of {correction_13} samples to synchronize devices 1 and 3.")
            synced_13 = True
            print("synchronized 1 and 3")

        else:
            print(
                f"Time difference detected between devices 1 and 3: {time_difference_13:.6f} seconds (less than TARGET_DIFFERENCE, no further adjustment)")

    # Beamforming
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)
        #delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c
        delays = distances * np.sin(steering_angle_rad) / c
        delay_samples = np.round(delays * RATE).astype(int)

        output_signal = np.zeros(num_samples)
        output_signal2 = np.zeros(num_samples)
        output_signal3 = np.zeros(num_samples)

        for i in range(CHANNELS * 3):
            delayed_signal = shift_signal(combined_signal[:, i], -delay_samples[i])
            delayed_signal2 = np.roll(combined_signal[:, i], -delay_samples[i])
            delayed_signal3 = np.roll(combined_signal2[:, i], -delay_samples[i])
            output_signal += delayed_signal # with shift
            output_signal2 += delayed_signal2 # with roll
            output_signal3 += delayed_signal3 # without synchronization
        output_signal /= (CHANNELS * 3)
        output_signal2 /= (CHANNELS * 3)
        output_signal3 /= (CHANNELS * 3)

        energy[idx] = np.sum(output_signal ** 2)
        energy2[idx] = np.sum(output_signal2 ** 2)
        energy3[idx] = np.sum(output_signal3 ** 2)

    estimated_angle = angles_range[np.argmax(energy)]
    angles_history.append(estimated_angle)

    line.set_ydata(energy)
    #line2.set_ydata(energy2)
    #line3.set_ydata(energy3)
    ax.set_ylim(0, max(energy) * 1.1)  # Ajustar dinámicamente el límite del eje y
    fig.canvas.draw()
    fig.canvas.flush_events()

# Detener los hilos de grabación después de que termine el procesamiento
stop_recording_and_save()

if angles_history:
    overall_avg_angle = np.mean(angles_history)
    print(f"Average angle over all windows: {overall_avg_angle:.2f} degrees")
else:
    print("No angles could be calculated in any window.")
