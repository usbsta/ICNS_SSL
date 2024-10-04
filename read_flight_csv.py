import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar los archivos CSV
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-07-49].csv'
file_path_flight = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-23-48].csv'

# Leer el archivo de referencia y de vuelo
ref_data = pd.read_csv(ref_file_path, skiprows=1, delimiter=',', low_memory=False)
flight_data = pd.read_csv(file_path_flight, skiprows=1, delimiter=',', low_memory=False)

# Extraer la posición de referencia (primer valor válido de latitud y longitud)
reference_latitude = ref_data['OSD.latitude'].dropna().astype(float).iloc[0]
reference_longitude = ref_data['OSD.longitude'].dropna().astype(float).iloc[0]

# Extraer las columnas necesarias: latitud, longitud, altura y tiempo
latitude_col = 'OSD.latitude'
longitude_col = 'OSD.longitude'
altitude_col = 'OSD.altitude [ft]'
time_col = 'CUSTOM.updateTime [local]'

# Filtrar las filas con datos válidos
flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()

# Convertir la altitud de pies a metros
flight_data[altitude_col] = flight_data[altitude_col] * 0.3048

# Obtener la altitud inicial del vuelo para usarla en la referencia de elevación
initial_altitude = flight_data[altitude_col].iloc[0]

# Convertir el tiempo a un formato adecuado para el gráfico
flight_data['Time'] = pd.to_datetime(flight_data[time_col], format='%I:%M:%S.%f %p')

# Función para calcular la distancia horizontal entre dos coordenadas
def calculate_horizontal_distance(lat1, lon1, lat2, lon2):
    # Convertir las coordenadas a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Radio de la Tierra en metros
    R = 6371000

    # Diferencias de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calcular la distancia usando la fórmula de Haversine
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Función para calcular azimuth
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    azimuth = np.degrees(np.arctan2(x, y))
    return azimuth

# Función para calcular elevación utilizando la distancia horizontal
def calculate_elevation(altitude, lat1, lon1, lat2, lon2, reference_altitude):
    horizontal_distance = calculate_horizontal_distance(lat1, lon1, lat2, lon2)
    relative_altitude = altitude - reference_altitude  # Calcular la altura relativa respecto a la referencia
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

# Calcular los valores iniciales de azimuth y elevación
initial_azimuth = calculate_azimuth(reference_latitude, reference_longitude,
                                    flight_data.iloc[0][latitude_col],
                                    flight_data.iloc[0][longitude_col])

initial_elevation = calculate_elevation(flight_data.iloc[0][altitude_col], reference_latitude, reference_longitude,
                                        flight_data.iloc[0][latitude_col], flight_data.iloc[0][longitude_col],
                                        initial_altitude)

# Configurar la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
point, = ax.plot([], [], 'bo', markersize=5)  # Crear un punto en lugar de una línea

ax.set_xlim(-180, 180)  # Rango en el eje X (azimuth en grados de -180 a 180)
ax.set_ylim(10, 90)  # Rango en el eje Y (elevación en grados de 0 a 90)
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-Time Drone Azimuth and Elevation')

# Función de actualización para la animación
# Función de actualización para la animación
def update(frame):
    lat = flight_data.iloc[frame][latitude_col]
    lon = flight_data.iloc[frame][longitude_col]
    altitude = flight_data.iloc[frame][altitude_col]

    # Calcular el azimuth y la elevación relativos al punto de referencia
    azimuth = calculate_azimuth(reference_latitude, reference_longitude, lat, lon) - initial_azimuth
    azimuth = -1 * (calculate_azimuth(reference_latitude, reference_longitude, lat, lon) - initial_azimuth)  # Invertir el ángulo de azimuth

    elevation = calculate_elevation(altitude, reference_latitude, reference_longitude, lat, lon, initial_altitude) - initial_elevation

    # Actualizar los datos del punto en el gráfico (como listas para evitar el error)
    point.set_data([azimuth], [elevation])  # Wrap azimuth and elevation in lists

    plt.draw()
    plt.pause(0.05)

# Animación en tiempo real
for i in range(len(flight_data)):
    update(i)

# Desactivar el modo interactivo
plt.ioff()
plt.show()
