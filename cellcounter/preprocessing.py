import numpy as np
import cv2

def load_and_gray(path):
    """Carga una imagen y la convierte a grises."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_frequency_filter(image, radius=75, filter_type='low'):
    """
    Filtro en frecuencia (Pasa-bajas o Pasa-altas).
    Ideal para limpiar el ruido de imágenes microscópicas.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Crear la máscara (tu lógica de clase)
    mask = np.zeros((rows, cols), np.float32)
    if filter_type == 'low':
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
    else: # Pasa-altas
        mask = np.ones((rows, cols), np.float32)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
        
    f_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    
    # Normalizar para que sea una imagen válida (0-255)
    return cv2.normalize(img_back, None, 255, cv2.NORM_MINMAX).astype(np.uint8)