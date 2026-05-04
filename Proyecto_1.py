import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Poder cargar la imgen con cualquier PC  es decir te encuentra la ruta de la imagen sin importar donde se ejecute el código, 
# esto es importante para que el código sea portable y pueda ser utilizado en diferentes entornos sin necesidad de modificar las rutas de los archivos.
# Al agregar la ruta del directorio actual al sys.path, se asegura que Python pueda encontrar y cargar los módulos necesarios 
# para el procesamiento de imágenes, independientemente de dónde se ejecute el script.

base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from cellcounter import preprocessing as cc_pre
from cellcounter import segmentation as cc_seg
from cellcounter import analysis as cc_ana
from cellcounter import hough as cc_hou


# FUNCIONES DE PROCESAMIENTO MANUAL Y GENERAL
#

def procesar_celulas_general(nombre_img, area_min=100, sens_hough=0.65):

    ruta_img = os.path.join(base_path, nombre_img)
    print(f"\n🔬 Procesando: {nombre_img}")
    
    # 1. Carga y Grises (Requerimiento Técnico)
    img = cc_pre.load_and_gray(ruta_img)
    
    # 2. Preprocesamiento (Filtros Espacial y Frecuencia)
    img_inc = cc_pre.apply_intensity_transform(img)
    # Filtro Mediana (Espacial)
    img_filt = cc_pre.spatial_filter(img_inc, filter_type='median', kernel_size=5)
    # Filtro FFT (Frecuencia)
    img_freq = cc_pre.frequency_filter(img_filt, radius=30, filter_type='low')
    
    # 3. Segmentación (Umbralización y Regiones)
    # Suavizado para una umbralización estable
    img_blur = cc_pre.spatial_filter(img_freq, filter_type='gaussian', kernel_size=7)
    img_bin = cc_seg.threshold_otsu(img_blur)
    
    # Relleno Manual (Lógica de Regiones)
    # Si un hueco negro está rodeado, se asume parte de la célula
    filas, cols = img_bin.shape
    for i in range(1, filas-1):
        for j in range(1, cols-1):
            if img_bin[i,j] == 0:
                if (np.any(img_bin[i, :j]) and np.any(img_bin[i, j:])):
                    img_bin[i,j] = 1

    # 4. Conteo Final
    total_celulas = cc_ana.count_cells(img_bin, min_area=area_min)
    
    # 5. Detección de Bordes y Hough
    bordes = cc_seg.detect_edges(img_bin, method='sobel')
    circulos_raw = cc_hou.hough_circles(bordes, min_radius=25, max_radius=115, threshold=sens_hough)
    
    # 6. Filtro de Proximidad Manual (NMS)
    circulos_finales = []
    if len(circulos_raw) > 0:
        circulos_raw = sorted(circulos_raw, key=lambda c: c[2], reverse=True)
        for c_nuevo in circulos_raw:
            es_unico = True
            for c_acc in circulos_finales:
                d = np.sqrt((c_nuevo[0]-c_acc[0])**2 + (c_nuevo[1]-c_acc[1])**2)
                if d < 60: # Distancia general de separación
                    es_unico = False
                    break
            if es_unico:
                circulos_finales.append(c_nuevo)
    
    return img, img_freq, img_bin, bordes, circulos_finales[:total_celulas], total_celulas

#
# EJECUCIÓN DEL PAQUETE
#

# Esta lógica permite procesar cualquier imagen enviando los parámetros

img_orig, img_f, img_b, bordes_l, lista_c, n_celulas = procesar_celulas_general(
    "celula1.jpg", area_min=80, sens_hough=0.6
)

# --- VISUALIZACIÓN MULTIPANEL ---
plt.figure(figsize=(16, 9))
plt.subplot(2,3,1); plt.imshow(img_orig, cmap='gray'); plt.title("1. Original")
plt.subplot(2,3,2); plt.imshow(img_f, cmap='gray'); plt.title("2. FFT (Frecuencia)")
plt.subplot(2,3,3); plt.imshow(img_b, cmap='gray'); plt.title(f"3. Segmentación ({n_celulas} obj)")
plt.subplot(2,3,4); plt.imshow(bordes_l, cmap='gray'); plt.title("4. Detección de Bordes")
plt.subplot(2,3,5); plt.imshow(img_orig, cmap='gray')
for x, y, r in lista_c:
    plt.gca().add_patch(plt.Circle((x, y), r, color='r', fill=False, linewidth=2))
plt.title(f"5. Hough ({len(lista_c)} círculos)")

plt.subplot(2,3,6); plt.axis('off')
plt.text(0.5, 0.5, f"REPORTE GENERAL\n\nCélulas: {n_celulas}\nCírculos: {len(lista_c)}", 
         ha='center', va='center', fontsize=14, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.show()