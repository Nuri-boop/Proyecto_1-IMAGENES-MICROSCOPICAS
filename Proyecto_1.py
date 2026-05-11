import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta de la carpeta actual
if base_path not in sys.path:
    sys.path.insert(0, base_path)  # Agrega la ruta al sistema para poder importar la librería

# Importación de los módulos de la librería propia
from cellcounter import preprocessing as cc_pre   # Módulo de preprocesamiento (filtros, intensidad)
from cellcounter import segmentation as cc_seg   # Módulo de segmentación (umbral, bordes, regiones)
from cellcounter import hough as cc_hou           # Módulo de transformada de Hough (círculos)

# ============================================================
# FUNCIÓN PRINCIPAL: PROCESA UNA IMAGEN Y CUENTA CÉLULAS
# ============================================================
def procesar_celulas_final(nombre_img):
    """Procesa una imagen microscópica y detecta células usando Hough."""
    
    # --------------------------------------------------------
    # 1. CARGA Y CONVERSIÓN A GRISES
    # --------------------------------------------------------
    ruta_img = os.path.join(base_path, nombre_img)  # Construye la ruta completa de la imagen
    img = cc_pre.load_and_gray(ruta_img)            # Carga la imagen y la convierte a escala de grises
    
    # --------------------------------------------------------
    # 2. PREPROCESAMIENTO: FILTROS ESPACIAL Y EN FRECUENCIA
    # --------------------------------------------------------
    # Filtro espacial: mediana para eliminar ruido (cada píxel se reemplaza por la mediana de sus vecinos)
    img_filt = cc_pre.spatial_filter(img, filter_type='median', kernel_size=5)
    
    # Filtro en frecuencia: pasa-bajas usando FFT para suavizar y eliminar detalles finos
    img_freq = cc_pre.frequency_filter(img_filt, radius=30, filter_type='low')
    
    # --------------------------------------------------------
    # 3. SEGMENTACIÓN: UMBRALIZACIÓN + CIERRE DE HUECOS
    # --------------------------------------------------------
    # Umbralización de Otsu: separa células (blanco) del fondo (negro) de forma automática
    img_bin = cc_seg.threshold_otsu(img_freq)
    
    # Cierre de huecos manual: rellena agujeros dentro de las células para que queden sólidas
    img_solida = cc_seg.cerrar_huecos_manual(img_bin, iteraciones=15)
    
    # --------------------------------------------------------
    # 4. DETECCIÓN DE BORDES (Edge Detection)
    # --------------------------------------------------------
    # Algoritmo de Sobel: detecta los contornos de las células
    bordes = cc_seg.detect_edges(img_solida, method='sobel')
    
    # --------------------------------------------------------
    # 5. TRANSFORMADA DE HOUGH PARA CÍRCULOS
    # --------------------------------------------------------
    # Detecta círculos en la imagen de bordes (las células suelen ser aproximadamente circulares)
    circulos_raw = cc_hou.hough_circles(bordes, min_radius=65, max_radius=100, threshold=0.65)
    
    # --------------------------------------------------------
    # 6. FILTRADO FINAL: ELIMINACIÓN DE FALSOS POSITIVOS
    # --------------------------------------------------------
    circulos_finales = []
    if len(circulos_raw) > 0:
        # Ordenar por radio (mayor a menor)
        circulos_raw = sorted(circulos_raw, key=lambda c: c[2], reverse=True)
        for x, y, r in circulos_raw:
            # Condición 1: el centro debe estar dentro de una célula (zona blanca)
            if img_solida[int(y), int(x)] == 255:
                # Condición 2: no debe haber otro círculo muy cerca (distancia mínima 110 píxeles)
                es_unico = True
                for c_acc in circulos_finales:
                    dist = np.sqrt((x - c_acc[0])**2 + (y - c_acc[1])**2)
                    if dist < 110: 
                        es_unico = False
                        break
                if es_unico:
                    circulos_finales.append((x, y, r))
    
    # Retorna: imagen original, filtrada, sólida, bordes, círculos detectados y el conteo
    return img, img_freq, img_solida, bordes, circulos_finales, len(circulos_finales)

# ============================================================
# EJECUCIÓN PRINCIPAL Y VISUALIZACIÓN DE RESULTADOS
# ============================================================
try:
    # Llamar a la función con la imagen de prueba "6.jpeg"
    img_o, img_f, img_s, bord_e, lista_c, conteo = procesar_celulas_final("4.jpg")

    # Crear una figura con 6 subgráficas (2 filas, 3 columnas)
    plt.figure(figsize=(15, 10))
    
    # Gráfica 1: Imagen original en escala de grises
    plt.subplot(2,3,1)
    plt.imshow(img_o, cmap='gray')
    plt.title("1. Original")
    plt.axis('off')
    
    # Gráfica 2: Imagen después del filtro en frecuencia (FFT)
    plt.subplot(2,3,2)
    plt.imshow(img_f, cmap='gray')
    plt.title("2. Filtro Frecuencia")
    plt.axis('off')
    
    # Gráfica 3: Imagen binarizada con células sólidas (rellenas)
    plt.subplot(2,3,3)
    plt.imshow(img_s, cmap='gray')
    plt.title("3. Segmentación Sólida")
    plt.axis('off')
    
    # Gráfica 4: Bordes detectados con Sobel
    plt.subplot(2,3,4)
    plt.imshow(bord_e, cmap='gray')
    plt.title("4. Bordes (Sobel)")
    plt.axis('off')
    
    # Gráfica 5: Imagen original con círculos detectados (resultado final)
    plt.subplot(2,3,5)
    plt.imshow(img_o, cmap='gray')
    for x, y, r in lista_c:   # Dibujar cada círculo detectado
        plt.gca().add_patch(plt.Circle((x, y), r, color='r', fill=False, linewidth=2))
    plt.title(f"5. Hough ({conteo} células)")
    plt.axis('off')
    
    # Gráfica 6: Reporte final con el número de células
    plt.subplot(2,3,6)
    plt.axis('off')
    plt.text(0.5, 0.5, f"REPORTE\n\nCélulas: {conteo}", 
             fontsize=20, fontweight='bold', ha='center', color='darkgreen')
    
    # Mostrar todas las gráficas
    plt.tight_layout()
    plt.show()

# Si ocurre algún error, mostrar el mensaje
except Exception as e:
    print(f"Error en el proceso: {e}")