import pyaudio
import sounddevice as sd

print(sd.query_devices())

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print (p.get_device_info_by_index(i))