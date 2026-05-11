import numpy as np

# ============================================================
# CONTEO DE CÉLULAS USANDO BFS (BÚSQUEDA EN ANCHURA)
# ============================================================
def count_cells(binary_image, min_area=100):
    """
    Cuenta el número de células en una imagen binaria usando BFS.
    
    PARÁMETROS:
    - binary_image: imagen binaria (0 = fondo NEGRO, 255 = célula BLANCO)
    - min_area: área mínima en píxeles para considerar una región como célula
                (evita contar ruido o regiones muy pequeñas)
    
    RETORNA:
    - cells_count: número de células detectadas
    
    ¿CÓMO FUNCIONA BFS?
    - Recorre la imagen píxel por píxel
    - Cuando encuentra un píxel blanco no visitado, comienza una exploración
    - Explora todos los píxeles blancos conectados (4 direcciones: arriba, abajo, izquierda, derecha)
    - Calcula el área del grupo encontrado
    - Si el área es suficiente, cuenta como una célula
    - Marca todos los píxeles del grupo como "visitados" para no volver a contarlos
    """
    
    # --------------------------------------------------------
    # PARTE 1: INICIALIZACIÓN
    # --------------------------------------------------------
    h, w = binary_image.shape          # Altura y ancho de la imagen
    visited = np.zeros_like(binary_image, dtype=bool)  # Matriz para marcar píxeles ya procesados
    cells_count = 0                    # Contador de células
    
    # --------------------------------------------------------
    # PARTE 2: RECORRER TODA LA IMAGEN
    # --------------------------------------------------------
    for i in range(h):      # Recorrer filas (Y)
        for j in range(w):  # Recorrer columnas (X)
            
            # BUSCAR UN PÍXEL BLANCO (CÉLULA) QUE NO HAYA SIDO VISITADO
            if binary_image[i, j] == 255 and not visited[i, j]:
                
                # --------------------------------------------------------
                # PARTE 3: BFS - EXPLORAR REGIÓN CONECTADA COMPLETA
                # --------------------------------------------------------
                area = 0                      # Iniciar área en 0
                queue = [(i, j)]              # Cola de píxeles por explorar
                visited[i, j] = True          # Marcar como visitado
                
                # Mientras haya píxeles en la cola...
                while queue:
                    curr_i, curr_j = queue.pop(0)  # Sacar el primer píxel de la cola
                    area += 1                       # Incrementar el área
                    
                    # Revisar los 4 VECINOS (arriba, abajo, izquierda, derecha)
                    # Esto es 4-conectividad (solo ejes, NO diagonales)
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni = curr_i + di   # Vecino fila
                        nj = curr_j + dj   # Vecino columna
                        
                        # Verificar que el vecino esté dentro de la imagen
                        if 0 <= ni < h and 0 <= nj < w:
                            
                            # Si el vecino es BLANCO y NO ha sido visitado
                            if binary_image[ni, nj] == 255 and not visited[ni, nj]:
                                visited[ni, nj] = True    # Marcar como visitado
                                queue.append((ni, nj))    # Agregar a la cola
                
                # --------------------------------------------------------
                # PARTE 4: DECIDIR SI ES UNA CÉLULA VÁLIDA
                # --------------------------------------------------------
                # Si el área del grupo supera el mínimo, contar como célula
                if area >= min_area:
                    cells_count += 1
                
                # Si el área es menor, se ignora (probablemente ruido)
    
    return cells_count