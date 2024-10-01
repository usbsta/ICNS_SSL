import numpy as np

d = 0.12
RATE = 48000
CHANNELS = 6
c = 343

mic_angles = np.array([0, 120, 240])
mic_angles = np.array([0, 90, 180, 270])
angles_range = np.arange(0, 361, 1)

delt=[] #time
dels=[] #samples

for angle in (angles_range):
    steering_angle_rad = np.radians(angle)

    delays = d * np.cos(np.radians(mic_angles) - steering_angle_rad) / c
    delays -= delays[0]  # Restar el retraso del primer micrófono (referencia)

    #delays = d * (np.cos(np.radians(mic_angles) - steering_angle_rad) - np.cos(steering_angle_rad)) / c

    delay_samples = np.round(delays * RATE).astype(int)
    delt.append(delays)
    dels.append(delay_samples)
    print(angle,delays)
    print(delay_samples)

print(delt[0],dels[0])
print(delt[90],dels[90])

