import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
file_path = '/Users/30068385/ICNS_SSL/Processed Data/3rd_Oct_11/beamforming_results.csv'
data = pd.read_csv(file_path)


# Crear una figura y subplots para los box plots
plt.figure(figsize=(10, 6))

# Box plot de la diferencia en azimut
plt.subplot(1, 2, 1)
plt.boxplot(data['Dif_Azimut'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title('Box Plot of Azimuth Difference')
plt.xlabel('Azimuth Difference (Degrees)')

# Box plot de la diferencia en elevación
plt.subplot(1, 2, 2)
plt.boxplot(data['Dif_Elevacion'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title('Box Plot of Elevation Difference')
plt.xlabel('Elevation Difference (Degrees)')

# Mostrar los box plots
plt.tight_layout()
plt.show()

# Crear una figura y subplots para los gráficos
plt.figure(figsize=(8, 6))

# Histograma de la diferencia en el azimut
plt.subplot(2, 2, 1)
plt.hist(data['Dif_Azimut'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Azimuth Difference')
plt.xlabel('Azimuth Difference (Degrees)')
plt.ylabel('Frequency')

# Histograma de la diferencia en la elevación
plt.subplot(2, 2, 2)
plt.hist(data['Dif_Elevacion'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogram of Elevation Difference')
plt.xlabel('Elevation Difference (Degrees)')
plt.ylabel('Frequency')

# Gráfico de dispersión de estimación vs. real de azimut
plt.subplot(2, 2, 3)
plt.scatter(data['Azimut_CSV'], data['Azimut_Estimado'], color='blue', alpha=0.5)
plt.title('Azimuth Estimation vs Real')
plt.xlabel('Real Azimuth (Degrees)')
plt.ylabel('Estimated Azimuth (Degrees)')
plt.grid(True)

# Gráfico de dispersión de estimación vs. real de elevación
plt.subplot(2, 2, 4)
plt.scatter(data['Elevacion_CSV'], data['Elevacion_Estimada'], color='green', alpha=0.5)
plt.title('Elevation Estimation vs Real')
plt.xlabel('Real Elevation (Degrees)')
plt.ylabel('Estimated Elevation (Degrees)')
plt.grid(True)

# Ajustar el diseño para evitar que se superpongan los gráficos
plt.tight_layout()

# Mostrar las gráficas
plt.show()

# Función para eliminar outliers usando el método IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Eliminar outliers en Dif_Azimut y Dif_Elevacion
data_clean = remove_outliers(data, 'Dif_Azimut')
data_clean = remove_outliers(data_clean, 'Dif_Elevacion')

# Crear una figura y subplots para los gráficos sin outliers
plt.figure(figsize=(8, 6))

# Histograma de la diferencia en azimut (sin outliers)
plt.subplot(2, 2, 1)
plt.hist(data_clean['Dif_Azimut'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Azimuth Difference (Without Outliers)')
plt.xlabel('Azimuth Difference (Degrees)')
plt.ylabel('Frequency')

# Histograma de la diferencia en elevación (sin outliers)
plt.subplot(2, 2, 2)
plt.hist(data_clean['Dif_Elevacion'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogram of Elevation Difference (Without Outliers)')
plt.xlabel('Elevation Difference (Degrees)')
plt.ylabel('Frequency')

# Gráfico de dispersión de estimación vs. real de azimut (sin outliers)
plt.subplot(2, 2, 3)
plt.scatter(data_clean['Azimut_CSV'], data_clean['Azimut_Estimado'], color='blue', alpha=0.5)
plt.title('Azimuth Estimation vs Real (Without Outliers)')
plt.xlabel('Real Azimuth (Degrees)')
plt.ylabel('Estimated Azimuth (Degrees)')
plt.grid(True)

# Gráfico de dispersión de estimación vs. real de elevación (sin outliers)
plt.subplot(2, 2, 4)
plt.scatter(data_clean['Elevacion_CSV'], data_clean['Elevacion_Estimada'], color='green', alpha=0.5)
plt.title('Elevation Estimation vs Real (Without Outliers)')
plt.xlabel('Real Elevation (Degrees)')
plt.ylabel('Estimated Elevation (Degrees)')
plt.grid(True)

# Ajustar el diseño para evitar que se superpongan los gráficos
plt.tight_layout()

# Mostrar las gráficas
plt.show()