import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt
from line_profiler import profile
from numba import njit, prange



@profile
def main():
    print('start calculating')


    CHANNELS = 6
    RATE = 48000
    CHUNK = int(0.1 * RATE)  # buffer 100 ms
    c = 343
    RECORD_SECONDS = 120000

    lowcut = 400.0
    highcut = 8000.0

    azimuth_range = np.arange(-180, 181, 5)
    elevation_range = np.arange(10, 91, 5)

    a = [0, -120, -240]
    # config 1 equidistance
    h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
    r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

    # config 2 augmented
    #h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
    #r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]


    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],  # Mic 6
        [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],  # Mic 7
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # Mic 8
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],  # Mic 9
        [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],  # Mic 10
        [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],  # Mic 11
        [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],  # Mic 12
        [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],  # Mic 13
        [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],  # Mic 14
        [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],  # Mic 15
        [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],  # Mic 16
        [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],  # Mic 17
        [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]] # Mic 18
    ])

    wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_1_sync.wav',
                     '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_2_sync.wav',
                     '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_3_sync.wav']


    buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

    # beamforming
    @profile
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

                delays = (np.dot(mic_positions, direction_vector) / c)

                # aplying delays
                output_signal = np.zeros(num_samples)
                for i, delay in enumerate(delays):
                    delay_samples = int(np.round(delay * RATE))
                    signal_shifted = np.roll(signal_data[:, i], delay_samples)
                    output_signal += signal_shifted

                output_signal /= signal_data.shape[1]  # normalize amplitud with num of mics
                energy[az_idx, el_idx] = np.sum(output_signal ** 2)
        return energy

    @profile
    def beamform_frequency(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
        num_samples = signal_data.shape[0]
        num_mics = signal_data.shape[1]
        num_freqs = num_samples // 2 + 1
        freqs = np.fft.rfftfreq(num_samples, d=1 / RATE)

        # FFT de las señales
        signal_fft = np.fft.rfft(signal_data, axis=0)

        # Pre-cálculo de las direcciones
        azimuth_rad = np.radians(azimuth_range)
        elevation_rad = np.radians(elevation_range)
        az_grid, el_grid = np.meshgrid(azimuth_rad, elevation_rad, indexing='ij')

        # Vector de dirección
        direction_vectors = np.stack([
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid)
        ], axis=2).reshape(-1, 3).T  # Shape: (3, num_directions)

        # Cálculo de los retrasos para todas las direcciones
        delays = mic_positions @ direction_vectors / c  # Shape: (num_mics, num_directions)

        # Cálculo de los cambios de fase
        phase_shifts = np.exp(-2j * np.pi * freqs[:, np.newaxis, np.newaxis] * delays[np.newaxis, :,
                                                                               :])  # Shape: (num_freqs, num_mics, num_directions)

        # Beamforming
        energy = beamform_frequency_numba(signal_fft, phase_shifts)

        # Reshape de la energía al tamaño original
        energy = energy.reshape(len(azimuth_range), len(elevation_range))

        return energy

    @njit(parallel=True)
    def beamform_frequency_numba(signal_fft, phase_shifts):
        num_freqs, num_mics = signal_fft.shape
        num_directions = phase_shifts.shape[2]
        energy = np.zeros(num_directions)

        for d in prange(num_directions):
            beamformed = np.zeros(num_freqs, dtype=np.complex64)
            for m in range(num_mics):
                beamformed += signal_fft[:, m] * phase_shifts[:, m, d]
            energy[d] = np.sum(np.abs(beamformed) ** 2)
        return energy

    def calculate_time(time_idx, chunk_size, rate):
        time_seconds = (time_idx * chunk_size) / rate
        return time_seconds

    def skip_wav_seconds(wav_file, seconds, rate):
        frames_to_skip = int(seconds * rate)
        wav_file.setpos(frames_to_skip)

    def butter_bandpass(lowcut, highcut, rate, order=5):
        nyquist = 0.5 * rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
        b, a = butter_bandpass(lowcut, highcut, rate, order=order)
        filtered_signal = filtfilt(b, a, signal_data, axis=0)  # Apply filter along the signal in each channel
        return filtered_signal

    def read_wav_block(wav_file, chunk_size):
        data = wav_file.readframes(chunk_size)
        if len(data) == 0:
            return None
        signal_data = np.frombuffer(data, dtype=np.int32)
        return np.reshape(signal_data, (-1, CHANNELS))

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Energy')
    ax.set_xlabel('Azimut')
    ax.set_ylabel('Elevation')
    ax.set_title('beamforming enery')
    max_energy_marker, = ax.plot([], [], 'ro')
    max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')


    wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

    skip_seconds = 115
    skip_seconds = 633

    for wav_file in wav_files:
        skip_wav_seconds(wav_file, skip_seconds, RATE)

    try:
        for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            finished = False # Flag to check if any file has finished

            # Read the next block of data for each device
            for i, wav_file in enumerate(wav_files):
                block = read_wav_block(wav_file, CHUNK)
                if block is None:
                    finished = True  # If one of the files reaches the end, activate the flag
                    break  # Break the loop if the end of the file is reached
                buffers[i] = block

            if finished:
                print("End of audio file.")
                break  # Exit the main loop if the end of the file is reached

            combined_signal = np.hstack(buffers)

            # filtering
            filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

            #energy = beamform_frequency(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
            energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

            # Find the index of the maximum energyce de la máxima energía
            max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
            estimated_azimuth = azimuth_range[max_energy_idx[0]]
            estimated_elevation = elevation_range[max_energy_idx[1]]

            # Calculate the current time of the audio sample
            current_time = calculate_time(time_idx, CHUNK, RATE)

            print(f"Time: {current_time + skip_seconds:.2f} s - Estimate angle: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

            cax.set_data(energy.T)
            cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))

            max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

            # Update the text with the coordinates
            max_energy_text.set_position((estimated_azimuth, estimated_elevation))
            max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

            fig.canvas.draw()
            fig.canvas.flush_events()

        print("Simulation completed.")

    finally:
        for wav_file in wav_files:
            wav_file.close()


if __name__ == '__main__':
    main()