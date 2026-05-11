import numpy as np

# ============================================================
# TRANSFORMADA DE HOUGH PARA DETECCIÓN DE CÍRCULOS
# ============================================================
def hough_circles(edge_image, min_radius=10, max_radius=45, threshold=0.3):
    """
    Detecta círculos en una imagen de bordes usando la Transformada de Hough.
    
    Parámetros:
    - edge_image: imagen binaria donde los bordes son blancos (255) y el fondo negro (0)
    - min_radius: radio mínimo esperado de los círculos (células pequeñas)
    - max_radius: radio máximo esperado de los círculos (células grandes)
    - threshold: umbral de votos mínimo (0 a 1). Ej: 0.3 = 30% del máximo de votos
    
    Retorna:
    - Lista de círculos detectados como (x, y, radio)
    """
    
    # --------------------------------------------------------
    # PARTE 1: REDIMENSIONAR LA IMAGEN PARA ACELERAR
    # --------------------------------------------------------
    h, w = edge_image.shape  # Altura y ancho de la imagen original
    
    scale = 1  # Factor de escala inicial
    if h > 250 or w > 250:  # Si la imagen es muy grande
        # Calcular escala para que el lado más grande sea 200 píxeles
        scale = min(200/h, 200/w)
        new_h, new_w = int(h*scale), int(w*scale)  # Nuevas dimensiones
        
        # Redimensionar manualmente (sin usar OpenCV)
        edge_small = np.zeros((new_h, new_w), dtype=edge_image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i/scale)  # Coordenada original en Y
                orig_j = int(j/scale)  # Coordenada original en X
                if 0 <= orig_i < h and 0 <= orig_j < w:
                    edge_small[i, j] = edge_image[orig_i, orig_j]
        h, w = new_h, new_w  # Actualizar dimensiones
    else:
        edge_small = edge_image  # Usar imagen original si es pequeña
    
    # --------------------------------------------------------
    # PARTE 2: ENCONTRAR TODOS LOS PUNTOS QUE SON BORDE
    # --------------------------------------------------------
    edge_points = []
    for i in range(h):
        for j in range(w):
            if edge_small[i, j] > 0:  # Si el píxel es blanco (borde)
                edge_points.append((i, j))  # Guardar sus coordenadas
    
    # Si no hay bordes, no hay círculos que detectar
    if len(edge_points) == 0:
        return []
    
    # --------------------------------------------------------
    # PARTE 3: ACUMULADOR DE VOTOS (ESPACIO x, y, radio)
    # --------------------------------------------------------
    accumulator = {}  # Diccionario: (x, y, radio) -> número de votos
    
    # Ajustar radios según la escala
    min_r = max(3, int(min_radius * scale))  # Radio mínimo en imagen redimensionada
    max_r = int(max_radius * scale)          # Radio máximo en imagen redimensionada
    radios = list(range(min_r, max_r + 1, 2))  # Probar radios de 2 en 2 para acelerar
    angulos = 16  # Número de ángulos a probar (cada 22.5 grados)
    
    # Para cada punto de borde...
    for y, x in edge_points:
        for r in radios:  # Probar diferentes radios
            for k in range(angulos):  # Probar diferentes ángulos
                theta = k * 2 * np.pi / angulos  # Ángulo en radianes
                # Fórmula del círculo: a = x - r*cos(theta), b = y - r*sin(theta)
                a = int(x - r * np.cos(theta))  # Posible coordenada X del centro
                b = int(y - r * np.sin(theta))  # Posible coordenada Y del centro
                
                # Si el centro propuesto está dentro de la imagen
                if 0 <= a < w and 0 <= b < h:
                    key = (a, b, r)  # Clave del acumulador
                    accumulator[key] = accumulator.get(key, 0) + 1  # Incrementar voto
    
    # Si no se acumularon votos, no hay círculos
    if not accumulator:
        return []
    
    # --------------------------------------------------------
    # PARTE 4: SELECCIONAR LOS CÍRCULOS CON MÁS VOTOS
    # --------------------------------------------------------
    max_votes = max(accumulator.values())   # Máximo número de votos obtenido
    min_votes = max_votes * threshold       # Voto mínimo requerido (ej: 30% del máximo)
    
    circles = []
    for (x, y, r), votes in accumulator.items():
        if votes >= min_votes:  # Si supera el umbral
            # Re-escalar coordenadas a la imagen original
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            r_orig = int(r / scale)
            
            # Evitar círculos muy cercanos entre sí (duplicados)
            is_unique = True
            for cx, cy, cr in circles:
                if abs(x_orig - cx) < 20 and abs(y_orig - cy) < 10:
                    is_unique = False
                    break
            
            # Agregar círculo si es único y radio cumple el mínimo
            if is_unique and r_orig >= min_radius:
                circles.append((x_orig, y_orig, r_orig))
    
    return circles


# ============================================================
# EROSIÓN MANUAL (OPERACIÓN MORFOLÓGICA)
# ============================================================
def manual_erosion(binary_image, iterations=1):
    """
    Aplica erosión a una imagen binaria.
    La erosión reduce el tamaño de los objetos blancos.
    
    Parámetros:
    - binary_image: imagen binaria (0 = fondo, 255 = objeto)
    - iterations: número de veces que aplicar la erosión
    
    Retorna:
    - Imagen erosionada
    """
    h, w = binary_image.shape
    eroded = binary_image.copy()
    
    for _ in range(iterations):  # Repetir N veces
        temp = eroded.copy()      # Copia para no afectar la iteración actual
        for i in range(1, h-1):   # Recorrer píxeles (evitando bordes)
            for j in range(1, w-1):
                # Ventana 3x3 alrededor del píxel
                neighborhood = temp[i-1:i+2, j-1:j+2]
                # Si algún vecino es negro (0), el píxel central se vuelve negro
                if np.any(neighborhood == 0):
                    eroded[i, j] = 0
    return eroded