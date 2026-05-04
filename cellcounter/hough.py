import numpy as np

def hough_circles(edge_image, min_radius=10, max_radius=45, threshold=0.3):
    """Transformada de Hough para detección de círculos (Hough transform)."""
    h, w = edge_image.shape
    
    # Reducir imagen para acelerar
    scale = 1
    if h > 250 or w > 250:
        scale = min(200/h, 200/w)
        new_h, new_w = int(h*scale), int(w*scale)
        edge_small = np.zeros((new_h, new_w), dtype=edge_image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i/scale)
                orig_j = int(j/scale)
                if 0 <= orig_i < h and 0 <= orig_j < w:
                    edge_small[i, j] = edge_image[orig_i, orig_j]
        h, w = new_h, new_w
    else:
        edge_small = edge_image
    
    # Encontrar puntos de borde
    edge_points = []
    for i in range(h):
        for j in range(w):
            if edge_small[i, j] > 0:
                edge_points.append((i, j))
    
    if len(edge_points) == 0:
        return []
    
    accumulator = {}
    min_r = max(3, int(min_radius*scale))
    max_r = int(max_radius*scale)
    radios = list(range(min_r, max_r + 1, 2))
    angulos = 16
    
    for y, x in edge_points:
        for r in radios:
            for k in range(angulos):
                theta = k * 2 * np.pi / angulos
                a = int(x - r * np.cos(theta))
                b = int(y - r * np.sin(theta))
                
                if 0 <= a < w and 0 <= b < h:
                    key = (a, b, r)
                    accumulator[key] = accumulator.get(key, 0) + 1
    
    if not accumulator:
        return []
    
    max_votes = max(accumulator.values())
    min_votes = max_votes * threshold
    
    circles = []
    for (x, y, r), votes in accumulator.items():
        if votes >= min_votes:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            r_orig = int(r / scale)
            
            is_unique = True
            for cx, cy, cr in circles:
                if abs(x_orig - cx) < 20 and abs(y_orig - cy) < 20:
                    is_unique = False
                    break
            if is_unique and r_orig >= min_radius:
                circles.append((x_orig, y_orig, r_orig))
    
    return circles