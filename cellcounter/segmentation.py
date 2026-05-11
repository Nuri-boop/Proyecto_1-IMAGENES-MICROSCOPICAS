import numpy as np

# ============================================================
# MÓDULO DE SEGMENTACIÓN
# ============================================================
# Este módulo contiene funciones para segmentar células,
# detectar bordes, aplicar umbralización y crecimiento de regiones.
# ============================================================

# ------------------------------------------------------------
# 1. CIERRE DE HUECOS MANUAL (OPERACIÓN MORFOLÓGICA)
# ------------------------------------------------------------
def cerrar_huecos_manual(img_bin, iteraciones=15):
    """
    Rellena huecos dentro de las células para que queden como masas sólidas.
    
    PARÁMETROS:
    - img_bin: imagen binaria (0 = fondo, 255 = célula/blanco)
    - iteraciones: número de veces que se repite el proceso
    
    ¿CÓMO FUNCIONA?
    - Recorre la imagen buscando píxeles NEGROS (0) que tengan algún vecino BLANCO (255)
    - Si encuentra uno, lo convierte a BLANCO
    - Así "rellena" agujeros dentro de las células
    - Se repite varias veces para rellenar completamente
    
    EJEMPLO VISUAL:
    Antes:           Después:
    ████████         ████████
    ██    ██   →     ████████
    ████████         ████████
    (hueco)          (relleno)
    """
    img_out = img_bin.copy()
    
    for _ in range(iteraciones):
        temp = img_out.copy()
        # Recorrer píxeles internos (evitando bordes)
        for i in range(1, img_out.shape[0]-1):
            for j in range(1, img_out.shape[1]-1):
                # Si el píxel es NEGRO (fondo/hueco)
                if temp[i, j] == 0:
                    # Verificar vecinos: ARRIBA, ABAJO, IZQUIERDA, DERECHA
                    if temp[i-1, j] == 255 or temp[i+1, j] == 255 or \
                       temp[i, j-1] == 255 or temp[i, j+1] == 255:
                        # Si algún vecino es blanco, rellenar
                        img_out[i, j] = 255
    return img_out


# ------------------------------------------------------------
# 2. UMBRALIZACIÓN DE OTSU (BINARIZACIÓN AUTOMÁTICA)
# ------------------------------------------------------------
def threshold_otsu(image):
    """
    Binarización automática usando el método de Otsu.
    Encuentra el umbral óptimo para separar células del fondo.
    
    ¿CÓMO FUNCIONA OTSU?
    - Asume que la imagen tiene DOS clases: fondo y objeto (células)
    - Prueba todos los umbrales posibles (0-255)
    - Calcula la "varianza entre clases" para cada uno
    - Elige el umbral que MAXIMIZA la varianza entre clases
    
    FÓRMULA: σ²_entre = w_fondo * w_objeto * (μ_fondo - μ_objeto)²
    
    RETORNA:
    - Imagen binaria: 255 (células) y 0 (fondo)
    """
    # Calcular histograma de intensidades (cuántos píxeles por cada valor 0-255)
    hist = np.bincount(image.ravel(), minlength=256)
    total = image.size  # Número total de píxeles
    sum_total = np.dot(np.arange(256), hist)  # Suma total de intensidades
    
    sum_b = 0      # Suma de intensidades del fondo
    w_b = 0        # Peso (cantidad de píxeles) del fondo
    var_max = 0    # Máxima varianza encontrada
    threshold = 0  # Umbral óptimo
    
    # Probar cada posible umbral (0 a 255)
    for i in range(256):
        w_b += hist[i]              # Actualizar peso del fondo
        if w_b == 0:
            continue
        
        w_f = total - w_b           # Peso del objeto (células)
        if w_f == 0:
            break
        
        sum_b += i * hist[i]        # Actualizar suma de intensidades del fondo
        m_b = sum_b / w_b           # Media del fondo
        m_f = (sum_total - sum_b) / w_f  # Media del objeto
        
        # Calcular varianza entre clases
        var_between = w_b * w_f * (m_b - m_f) ** 2
        
        # Si esta varianza es la mayor hasta ahora, guardar el umbral
        if var_between > var_max:
            var_max = var_between
            threshold = i
    
    # Aplicar umbral: píxeles > umbral se convierten en 255 (células)
    return (image > threshold).astype(np.uint8) * 255


# ------------------------------------------------------------
# 3. DETECCIÓN DE BORDES (SOBEL)
# ------------------------------------------------------------
def detect_edges(image, method='sobel'):
    """
    Detecta bordes en la imagen usando el operador Sobel.
    
    ¿CÓMO FUNCIONA SOBEL?
    - Usa dos kernels (máscaras) para detectar gradientes:
      - Gx: detecta cambios HORIZONTALES (bordes verticales)
      - Gy: detecta cambios VERTICALES (bordes horizontales)
    - Magnitud del borde = sqrt(Gx² + Gy²)
    - Valores altos = cambios bruscos = bordes
    
    KERNELS SOBEL:
    Kx (horizontal):     Ky (vertical):
    [-1, 0, 1]          [-1,-2,-1]
    [-2, 0, 2]          [ 0, 0, 0]
    [-1, 0, 1]          [ 1, 2, 1]
    """
    # Kernels de Sobel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    h, w = image.shape
    edges = np.zeros((h, w))
    
    # Recorrer cada píxel (excepto bordes)
    for i in range(1, h-1):
        for j in range(1, w-1):
            # Extraer ventana 3x3 alrededor del píxel
            region = image[i-1:i+2, j-1:j+2]
            
            # Calcular gradientes con convolución
            gx = np.sum(region * Kx)  # Gradiente horizontal
            gy = np.sum(region * Ky)  # Gradiente vertical
            
            # Magnitud del borde
            edges[i, j] = np.sqrt(gx**2 + gy**2)
    
    # Umbralizar: solo mantener bordes más fuertes que 100
    return (edges > 100).astype(np.uint8) * 255


# ------------------------------------------------------------
# 4. SEGMENTACIÓN POR CRECIMIENTO DE REGIONES (REGION-BASED)
# ------------------------------------------------------------
def region_growing(image, threshold=15):
    """
    Segmentación por crecimiento de regiones.
    Agrupa píxeles similares en regiones conectadas.
    
    PARÁMETROS:
    - image: imagen en escala de grises
    - threshold: tolerancia para incluir píxeles similares (0-255)
    
    ¿CÓMO FUNCIONA?
    1. Busca un píxel no visitado que no sea fondo (intensidad > 50)
    2. Lo usa como "semilla" para comenzar una región
    3. Agrega vecinos que tengan intensidad similar (diferencia ≤ threshold)
    4. Repite hasta que la región no crece más
    5. Si la región tiene área > 50 píxeles, se conserva como célula
    6. Si es más pequeña, se descarta (ruido)
    
    RETORNA:
    - Matriz con etiquetas de región (0 = fondo, 1,2,3... = células)
    """
    h, w = image.shape
    segmented = np.zeros((h, w), dtype=np.int32)  # 0 = fondo/no visitado
    region_id = 1
    
    for i in range(h):
        for j in range(w):
            # Buscar píxel NO VISITADO que NO sea fondo (intensidad > 50)
            if segmented[i, j] == 0 and image[i, j] > 50:
                
                # INICIAR NUEVA REGIÓN
                queue = [(i, j)]           # Cola de píxeles por explorar
                segmented[i, j] = region_id  # Marcar como parte de la región
                pixels = [(i, j)]          # Lista de píxeles en la región
                intensidad_original = image[i, j]  # Intensidad de la semilla
                
                # CRECIMIENTO DE REGIÓN (BFS)
                while queue:
                    y, x = queue.pop(0)  # Sacar píxel de la cola
                    
                    # Revisar 4 vecinos (arriba, abajo, izquierda, derecha)
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y + dy, x + dx
                        
                        if 0 <= ny < h and 0 <= nx < w:
                            # Si el vecino NO ha sido visitado
                            if segmented[ny, nx] == 0:
                                # CRITERIO DE SIMILITUD: diferencia de intensidad
                                diferencia = abs(int(image[ny, nx]) - intensidad_original)
                                
                                if diferencia <= threshold:
                                    # Es similar → pertenece a la misma región
                                    segmented[ny, nx] = region_id
                                    queue.append((ny, nx))
                                    pixels.append((ny, nx))
                
                # VALIDAR REGIÓN: ¿es una célula o es ruido?
                if len(pixels) > 50:  # Área mínima de 50 píxeles
                    region_id += 1    # Conservar región, pasar a la siguiente
                else:
                    # Región muy pequeña → es ruido → borrar
                    for y, x in pixels:
                        segmented[y, x] = 0
    
    return segmented