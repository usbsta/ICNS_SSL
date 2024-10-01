import pyaudio
import wave
import threading

FORMAT = pyaudio.paInt24  # 24 bits
CHANNELS = 6
RATE = 48000 # sample rate
CHUNK = 1024  # buffer size
RECORD_SECONDS = 2


device_index_1 = 1
device_index_2 = 2

audio = pyaudio.PyAudio()


def record_device(device_index, output_filename):

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    print(f"Recording from device {device_index}...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(f"Finished recording from device {device_index}.")

    stream.stop_stream()
    stream.close()

    # Save .wav
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Save as: {output_filename}")

thread_1 = threading.Thread(target=record_device, args=(device_index_1, "output_Pyaudiodevice_1.wav"))
thread_2 = threading.Thread(target=record_device, args=(device_index_2, "output_Pyaudiodevice_2.wav"))

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()

audio.terminate()

print("Completed recording from devices.")
