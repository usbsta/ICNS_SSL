import matplotlib.pyplot as plt
import numpy as np

# Dimensiones de la grilla
azimuth_range = np.arange(-180, 181, 10)  # Rango de azimuth: -180 a 180 con pasos de 10
elevation_range = np.arange(0, 91, 10)    # Rango de elevación: 0 a 90 con pasos de 10

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 6))  # Ajusta el tamaño según tus necesidades
ax.set_xlim(-180, 180)  # Límite del eje X (azimuth)
ax.set_ylim(0, 90)      # Límite del eje Y (elevación)

# Dibujar la grilla
ax.grid(True, which='both', linestyle='-', color='black', linewidth=1)

# Ajuste de las marcas en ambos ejes
ax.set_xticks(azimuth_range)  # Marcas en el eje X
ax.set_yticks(elevation_range)  # Marcas en el eje Y

# Remover el marco y los ejes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticklabels([])  # Quitar etiquetas del eje X
ax.set_yticklabels([])  # Quitar etiquetas del eje Y

# Ajustar fondo transparente
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Guardar la imagen como PNG con transparencia
plt.savefig('grid_overlay.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)

# Mostrar la imagen (opcional)
plt.show()
