#polar plot, el que mejor funciona
import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys

FORMAT = pyaudio.paInt32  # 32 bits
CHANNELS = 3
RATE = 48000
CHUNK = int(0.2 * RATE)
d = 0.12  # radio
c = 343
RECORD_SECONDS = 1200000
TARGET_DIFFERENCE = 200e-6
peak_threshold = 0.6e9

device_index_1 = 1  # Zoom 3
#device_index_2 = 4  # Zoom 2
#device_index_3 = 5  # Zoom

audio = pyaudio.PyAudio()


mic_angles = np.array([0, 120, 240])
angles_range = np.arange(0, 361, 1)

#output_filenames = ['device_1_sync.wav', 'device_2_sync.wav', 'device_3_sync.wav']

plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})  # Gráfico polar para azimut
line, = ax.plot(np.radians(angles_range), np.zeros(len(angles_range)), label='Azimuth Energy')
ax.set_ylim(0, 1)
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Energy')
plt.legend(loc="lower right")

buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]  # syncro
frames = [[], [], []]  # frames for 3 devices

'''
correction_12 = None  # correction between 1 and 2
correction_13 = None

total_correction_12 = 0
total_correction_13 = 0

synced_12 = False
synced_13 = False

window_size = int(0.05 * RATE)
'''

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


def record_device(device_index, buffer_index, stop_event):
    #global correction_12, correction_13, total_correction_12, total_correction_13

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

                '''
                # Aplica correcciones acumuladas
                if buffer_index == 1 and total_correction_12 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_12)

                if buffer_index == 2 and total_correction_13 != 0:
                    buffers[buffer_index] = shift_signal(buffers[buffer_index], total_correction_13)    

                frames[buffer_index].append(buffers[buffer_index].tobytes())
                '''

    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        print(f"Stream closed for device {device_index}")

'''
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

    audio.terminate()
    print("Recording saved and audio terminated.")
'''

# Iniciar grabación
threads = []
threads.append(threading.Thread(target=record_device, args=(device_index_1, 0, stop_event)))
#threads.append(threading.Thread(target=record_device, args=(device_index_2, 1, stop_event)))
#threads.append(threading.Thread(target=record_device, args=(device_index_3, 2, stop_event)))

for thread in threads:
    thread.start()

for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    combined_signal = np.hstack(buffers)
    num_samples = combined_signal.shape[0]
    energy = np.zeros(len(angles_range))

    # azimuth calculation 360
    for idx, angle in enumerate(angles_range):
        steering_angle_rad = np.radians(angle)

        #delays = d * (np.cos(np.radians(mic_angles) - steering_angle_rad) - np.cos(steering_angle_rad)) / c

        delays = d * np.cos(np.radians(mic_angles) - steering_angle_rad) / c
        delays -= delays[0]  # Restar el retraso del primer micrófono (referencia)

        delay_samples = np.round(delays * RATE).astype(int)
        output_signal = np.zeros(num_samples)

        for i in range(3):
            delayed_signal = np.roll(combined_signal[:, i], -delay_samples[i])
            output_signal += delayed_signal

        output_signal /= 3
        energy[idx] = np.sum(output_signal ** 2)

    estimated_angle = angles_range[np.argmax(energy)]  # Ángulo con mayor energía
    print(estimated_angle)
    # Actualizar el gráfico de azimut
    line.set_ydata(energy)
    ax.set_ylim(0, max(energy) * 1.1)  # Ajustar dinámicamente el límite del eje y
    fig.canvas.draw()
    fig.canvas.flush_events()

# Detener los hilos de grabación después de que termine el procesamiento
#stop_recording_and_save()
