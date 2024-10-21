import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyproj import Transformer
import pandas as pd

# Configuración de los datos y transformaciones
transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)  # Zona UTM 56 South

# Archivos CSV de vuelo y referencia
flight_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-10-11_[15-49-12].csv'
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-10-11_[14-32-34].csv'

# Leer los archivos CSV
flight_data = pd.read_csv(flight_file_path, skiprows=1, delimiter=',', low_memory=False)
ref_data = pd.read_csv(ref_file_path, skiprows=1, delimiter=',', low_memory=False)

# Extraer las columnas necesarias
latitude_col = 'OSD.latitude'
longitude_col = 'OSD.longitude'
altitude_col = 'OSD.altitude [ft]'

# Procesar los datos del vuelo
flight_data = flight_data[[latitude_col, longitude_col, altitude_col]].dropna()
flight_data[altitude_col] = flight_data[altitude_col] * 0.3048  # Convertir pies a metros

# Procesar los datos de referencia y calcular la posición promedio
ref_latitude = ref_data[latitude_col].dropna().astype(float).mean()
ref_longitude = ref_data[longitude_col].dropna().astype(float).mean()
ref_altitude = ref_data[altitude_col].dropna().astype(float).mean() * 0.3048  # Convertir a metros

# Convertir coordenadas geográficas a UTM
flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
    flight_data[longitude_col].values,
    flight_data[latitude_col].values
)

ref_x, ref_y = transformer.transform(ref_longitude, ref_latitude)

# Inicialización de la figura
fig, ax = plt.subplots()
trajectory_line, = ax.plot([], [], 'b-', label='Trajectory')
current_position, = ax.plot([], [], 'k+', markersize=20, label='Drone')
ref_position, = ax.plot([ref_x], [ref_y], 'g*', markersize=15, label='Mic Array')

ax.set_xlim(flight_data['X_meters'].min() - 10, flight_data['X_meters'].max() + 10)
ax.set_ylim(flight_data['Y_meters'].min() - 10, flight_data['Y_meters'].max() + 10)
ax.set_xlabel('X (mts)')
ax.set_ylabel('Y (mts)')
ax.set_title('Drone Position')
ax.legend()

# Función para calcular la distancia total (2D + altitud)
def calculate_total_distance(x1, y1, x2, y2, z1, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

# Función de actualización para la animación
def update(frame):
    x = flight_data.iloc[frame]['X_meters']
    y = flight_data.iloc[frame]['Y_meters']
    altitude = flight_data.iloc[frame][altitude_col]

    # Actualizar la trayectoria y la posición actual
    trajectory_line.set_data(flight_data['X_meters'][:frame+1], flight_data['Y_meters'][:frame+1])
    current_position.set_data([x], [y])

    # Calcular la distancia total y mostrarla en el título
    distance = calculate_total_distance(ref_x, ref_y, x, y, ref_altitude, altitude)
    ax.set_title(f'Distance to Ref: {distance:.2f} mts')

    return trajectory_line, current_position

# Animación en tiempo real, que se ejecuta solo una vez
ani = FuncAnimation(
    fig, update, frames=len(flight_data), interval=100, blit=False, repeat=False
)  # repeat=False asegura que solo se ejecute una vez

plt.show()