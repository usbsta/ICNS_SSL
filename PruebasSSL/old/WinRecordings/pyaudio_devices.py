import pyaudio

# Inicializar PyAudio
audio = pyaudio.PyAudio()

# Enumerar todos los dispositivos disponibles
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']} - {device_info['maxInputChannels']} input channels")

# Terminar PyAudio
audio.terminate()

