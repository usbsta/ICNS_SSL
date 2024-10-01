import numpy as np

d = 0.12
RATE = 48000
#CHANNELS = 6
c = 343

mic_positions = np.array([
    [0.12 * np.cos(np.radians(0)), 0.12 * np.sin(np.radians(0)), 0.5],  # Mic 1
    [0.12 * np.cos(np.radians(120)), 0.12 * np.sin(np.radians(120)), 0.5],  # Mic 2
    [0.12 * np.cos(np.radians(240)), 0.12 * np.sin(np.radians(240)), 0.5],  # Mic 3
    [0.2 * np.cos(np.radians(0)), 0.2 * np.sin(np.radians(0)), 0.25],  # Mic 4
    [0.2 * np.cos(np.radians(120)), 0.2 * np.sin(np.radians(120)), 0.25],  # Mic 5
    [0.2 * np.cos(np.radians(240)), 0.2 * np.sin(np.radians(240)), 0.25]  # Mic 6
])


azimuth_range = np.arange(0, 361, 10)  # Ángulos de azimut de -90° a 90°
elevation_range = np.arange(0, 90, 10)  # Ángulos de elevación de -90° a 90°
#elevation_range = np.arange(0, 91, 2)

delt=[] #time
dels=[] #samples

signal_data = np.linspace(-np.pi, np.pi, 48000)
num_samples = signal_data.shape[0]

ref_mic_position = mic_positions[0]
for az_idx, theta in enumerate(azimuth_range):
    azimuth_rad = np.radians(theta)

    for el_idx, phi in enumerate(elevation_range):
        elevation_rad = np.radians(phi)

        # Vector de dirección en 3D
        direction_vector = np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad)
        ])

        # Cálculo de los retrasos en tiempo para cada micrófono con respecto al micrófono 1
        ref_delay = np.dot(ref_mic_position, direction_vector) / c
        delays = (np.dot(mic_positions, direction_vector) / c) - ref_delay
        #delay_samples = int(np.round(delays * RATE))
        output_signal = np.zeros(num_samples)


        for i, delay in enumerate(delays):
           delay_samples = int(np.round(delay * RATE))  # Convertir el retraso en muestras
           delt.append(delays)
           dels.append(delay_samples)

           if delay_samples > 0:
               #output_signal[:num_samples - delay_samples] += signal_data[delay_samples:, i]
               output_signal[:num_samples - delay_samples] += signal_data[delay_samples:]
           else:
               #output_signal[-delay_samples:] += signal_data[:num_samples + delay_samples, i]
               output_signal[-delay_samples:] += signal_data[:num_samples + delay_samples]


    #print(angle,delays)
    #print(delay_samples)


#print(delt[0],dels[0])
#print(delt[90],dels[90])
print("Deltas:", delt[:5])
print("Delays en muestras:", dels[:5])

