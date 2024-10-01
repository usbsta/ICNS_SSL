import numpy as np

d = 0.04
RATE = 48000
CHANNELS = 6
c = 343

angles_range = np.arange(-90, 91, 1)
delt=[] #time
dels=[] #samples

for angle in (angles_range):
    steering_angle_rad = np.radians(angle)
    delays = np.arange(CHANNELS * 3) * d * np.sin(steering_angle_rad) / c
    delay_samples = np.round(delays * RATE).astype(int)
    delt.append(delays)
    dels.append(delay_samples)
    print(angle,delays)
    print(delay_samples)

print(delt[0],dels[0])
print(delt[90],dels[90])