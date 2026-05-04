import numpy as np

def count_cells(binary_image, min_area=100):
    """
    Conteo de células usando BFS 

    """
    h, w = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    cells_count = 0
    
    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 255 and not visited[i, j]:
                area = 0
                queue = [(i, j)]
                visited[i, j] = True
                
                while queue:
                    curr_i, curr_j = queue.pop(0)
                    area += 1
                    
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = curr_i + di, curr_j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if binary_image[ni, nj] == 255 and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                
                if area >= min_area:
                    cells_count += 1
    
    return cells_count