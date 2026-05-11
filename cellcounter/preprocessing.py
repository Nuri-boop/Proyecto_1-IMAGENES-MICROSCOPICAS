import numpy as np
import cv2

# ============================================================
# MÓDULO DE PREPROCESAMIENTO
# ============================================================
# Este módulo contiene funciones para preparar las imágenes
# antes de la segmentación y detección de células.
# ============================================================

# ------------------------------------------------------------
# 1. CARGA Y CONVERSIÓN A ESCALA DE GRISES
# ------------------------------------------------------------
def load_and_gray(path):
    """
    Carga una imagen desde un archivo y la convierte a escala de grises.
    
    PARÁMETROS:
    - path: ruta del archivo de imagen (ej: "celula.jpg")
    
    RETORNA:
    - Imagen en escala de grises (matriz 2D de valores 0-255)
    
    ¿POR QUÉ CONVERTIR A GRISES?
    - Las imágenes a color tienen 3 canales (R,G,B) = más datos
    - Escala de grises = 1 canal = menos cómputo
    - La información de forma/borde es suficiente para detectar células
    """
    img = cv2.imread(path)  # Leer imagen con OpenCV
    if img is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir BGR a grises


# ------------------------------------------------------------
# 2. TRANSFORMACIÓN DE INTENSIDAD (ESTIRAMIENTO DE CONTRASTE)
# ------------------------------------------------------------
def apply_intensity_transform(image):
    """
    Aplica estiramiento de contraste (Min-Max Scaling).
    Expande el rango de intensidades para mejorar el contraste.
    
    EJEMPLO:
    - Imagen original: píxeles entre 50 y 200
    - Transformación: se re-escalan a 0-255
    - Resultado: más contraste entre células y fondo
    
    FÓRMULA: nuevo_valor = (original - min) * 255 / (max - min)
    """
    img_min = np.min(image)  # Valor mínimo de intensidad en la imagen
    img_max = np.max(image)  # Valor máximo de intensidad en la imagen
    
    # Si todos los píxeles son iguales, no hay nada que estirar
    if img_max == img_min:
        return image
    
    # Escalar linealmente al rango [0, 255]
    rescaled = (image - img_min) * (255.0 / (img_max - img_min))
    return rescaled.astype(np.uint8)  # Convertir a enteros 0-255


# ------------------------------------------------------------
# 3. FILTRADO EN FRECUENCIA (USANDO FFT)
# ------------------------------------------------------------
def frequency_filter(image, radius=30, filter_type='low'):
    """
    Filtro en el dominio de la frecuencia usando Transformada Rápida de Fourier (FFT).
    
    PARÁMETROS:
    - image: imagen de entrada
    - radius: radio del filtro (cuántas frecuencias se conservan)
    - filter_type: 'low' (pasa-bajas) o 'high' (pasa-altas)
    
    ¿CÓMO FUNCIONA?
    - FFT convierte la imagen al dominio de frecuencia
    - Frecuencias bajas = cambios suaves (células completas)
    - Frecuencias altas = cambios bruscos (ruido, bordes finos)
    - Filtro pasa-bajas: conserva frecuencias bajas, elimina altas (ruido)
    - Filtro pasa-altas: conserva bordes, elimina suavizado
    """
    # PASO 1: Aplicar FFT 2D
    f = np.fft.fft2(image)           # Transformada de Fourier
    fshift = np.fft.fftshift(f)      # Mover frecuencias bajas al centro
    
    # PASO 2: Crear máscara circular
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Centro de la imagen
    
    # Crear cuadrícula de coordenadas
    y, x = np.ogrid[:rows, :cols]
    # Máscara circular: dentro del radio = True, fuera = False
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    
    # PASO 3: Elegir tipo de filtro
    if filter_type == 'low':
        mask = mask_area.astype(float)      # Conservar frecuencias CENTRALES (bajas)
    elif filter_type == 'high':
        mask = (~mask_area).astype(float)   # Conservar frecuencias EXTERIORES (altas)
    else:
        raise ValueError("filter_type debe ser 'low' o 'high'")
    
    # PASO 4: Aplicar filtro en frecuencia
    f_filt = fshift * mask                # Multiplicar por la máscara
    
    # PASO 5: Transformada inversa para volver al dominio espacial
    img_back = np.fft.ifft2(np.fft.ifftshift(f_filt))
    return np.abs(img_back).astype(np.uint8)  # Valor absoluto y convertir a uint8


# ------------------------------------------------------------
# 4. FILTRADO ESPACIAL (MEDIANA, MEDIA, GAUSSIANO)
# ------------------------------------------------------------
def spatial_filter(image, filter_type='median', kernel_size=3):
    """
    Aplica filtros espaciales directamente sobre los píxeles de la imagen.
    
    PARÁMETROS:
    - image: imagen de entrada
    - filter_type: 'median', 'mean', o 'gaussian'
    - kernel_size: tamaño de la ventana (3, 5, 7, etc.)
    
    ¿QUÉ HACE CADA UNO?
    - MEDIANA: Reemplaza cada píxel por la mediana de sus vecinos.
              EXCELENTE para eliminar ruido "sal y pimienta".
    - MEDIA:   Reemplaza por el promedio. Suaviza pero también borra bordes.
    - GAUSSIANO: Promedio ponderado (pesos según distancia al centro).
                Suavizado más natural.
    """
    h, w = image.shape
    pad = kernel_size // 2  # Cantidad de píxeles a añadir en cada borde
    
    # Rellenar bordes con ceros para poder aplicar el filtro en los bordes
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)
    
    # --------------------------------------------------------
    # FILTRO MEDIANA
    # --------------------------------------------------------
    if filter_type == 'median':
        for i in range(h):               # Para cada fila
            for j in range(w):           # Para cada columna
                # Extraer ventana de tamaño kernel_size x kernel_size
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(window)  # Mediana de la ventana
    
    # --------------------------------------------------------
    # FILTRO MEDIA
    # --------------------------------------------------------
    elif filter_type == 'mean':
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.mean(window)  # Promedio de la ventana
    
    # --------------------------------------------------------
    # FILTRO GAUSSIANO (crea kernel con distribución gaussiana)
    # --------------------------------------------------------
    elif filter_type == 'gaussian':
        sigma = 1.0  # Desviación estándar (controla el "suavizado")
        
        # CREAR KERNEL GAUSSIANO
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                # Fórmula de Gaussiana: e^(-(x²+y²)/(2σ²))
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)  # Normalizar (suma = 1)
        
        # APLICAR CONVOLUCIÓN
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.sum(window * kernel)  # Suma ponderada
    
    return output.astype(np.uint8)  # Convertir a uint8 (0-255)