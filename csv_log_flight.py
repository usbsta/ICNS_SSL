
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar el archivo CSV
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/DJIFlightRecord_2024-09-24_[13-07-49].csv'
file_path = '/Users/30068385/OneDrive - Western Sydney University/DJIFlightRecord_2024-09-19_[17-02-43].csv'

# Leer el archivo e ignorar la primera línea
data = pd.read_csv(file_path, skiprows=1, delimiter=',')

# Parámetros del observador
observer_distance = 8  # Distancia del observador al origen en metros

# Extraer los datos relevantes: latitud, longitud, y altitud
latitude = data['OSD.latitude'].dropna().astype(float)
longitude = data['OSD.longitude'].dropna().astype(float)
#altitude_ft = data['OSD.altitude [ft]'].dropna().astype(float)
altitude_ft = data['OSD.height [ft]'].dropna().astype(float)

# Convertir altitud de pies a metros
altitude_m = altitude_ft * 0.3048

initial_altitude = altitude_m.iloc[0]
print(initial_altitude)
altitude_m_corrected = altitude_m - initial_altitude

# Calcular la distancia horizontal en metros a partir de los cambios de latitud y longitud
# Consideramos la primera posición del dron como el origen
lat0, lon0 = latitude.iloc[0], longitude.iloc[0]
R = 6371000  # Radio de la Tierra en metros

# Calcular desplazamiento en metros
delta_lat = np.radians(latitude - lat0)
delta_lon = np.radians(longitude - lon0)
a = np.sin(delta_lat / 2)**2 + np.cos(np.radians(lat0)) * np.cos(np.radians(latitude)) * np.sin(delta_lon / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance_horizontal = R * c

# Calcular el ángulo de azimuth (en grados)
azimuth = np.degrees(np.arctan2(delta_lon, delta_lat))

# Calcular el ángulo de elevación (en grados) desde el observador
elevation_angle = np.degrees(np.arctan(altitude_m_corrected / observer_distance))

# Filtrar para valores válidos de elevación
valid_indices = (elevation_angle >= 5) & (elevation_angle <= 90)
azimuth = azimuth[valid_indices]
elevation_angle = elevation_angle[valid_indices]

# Configurar la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))

ax.set_ylim([0, 90])  # Limitar el ángulo de elevación entre 5 y 90 grados

# Crear el gráfico inicial
scat = ax.scatter(np.radians(azimuth), elevation_angle)

# Actualizar la gráfica en tiempo real
for i in range(len(azimuth)):
    scat.set_offsets(np.c_[np.radians(azimuth[:i+1]), elevation_angle[:i+1]])
    plt.draw()
    plt.pause(0.05)

# Desactivar el modo interactivo
plt.ioff()
plt.show()

