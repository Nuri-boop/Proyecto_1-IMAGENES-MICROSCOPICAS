import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONES DE LA LIBRERÍA (PROPIAS)
# ==========================================

def rgb_a_gris(img_rgb):
    """Convierte RGB a escala de grises manualmente."""
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def transformacion_log(img):
    """Mejora el contraste (Transformación de Intensidad)."""
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    return log_img.astype(np.uint8)

def filtro_frecuencia_pasa_bajos(img):
    """Suaviza la imagen usando la Transformada de Fourier."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    filas, cols = img.shape
    crow, ccol = filas // 2, cols // 2
    
    # Máscara circular (Pasa-Bajos)
    mask = np.zeros((filas, cols), np.uint8)
    radio = 30 
    center = [crow, ccol]
    x, y = np.ogrid[:filas, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= radio**2
    mask[mask_area] = 1
    
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back).astype(np.uint8)

def detector_sobel(img):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    img_x = cv2.filter2D(img, -1, gx)
    img_y = cv2.filter2D(img, -1, gy)
    
    sobel = np.sqrt(img_x**2 + img_y**2)
    
    # Evitamos la división por cero
    max_val = sobel.max()
    if max_val == 0:
        return sobel.astype(np.uint8)
        
    return (sobel / max_val * 255).astype(np.uint8)

def umbralizacion_manual(img, valor):
    """Convierte a blanco y negro puro."""
    binaria = np.zeros_like(img)
    binaria[img > valor] = 255
    return binaria

def contar_celulas(img_binaria):
    """Cuenta las regiones blancas (objetos)."""
    # connectedComponents cuenta grupos de píxeles blancos conectados
    num_labels, labels = cv2.connectedComponents(img_binaria)
    return num_labels - 1

# ==========================================
# 2. SCRIPT DE PRUEBA (EJECUCIÓN)
# ==========================================

def ejecutar_proyecto(nombre_imagen):
    # Leer imagen
    img_bgr = cv2.imread(nombre_imagen)
    if img_bgr is None:
        print("Error: No se encontró la imagen. Revisa el nombre.")
        return
    
    # 1. Proceso
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gris = rgb_a_gris(img_rgb)
    log = transformacion_log(gris)
    suave = filtro_frecuencia_pasa_bajos(log)
    bordes = detector_sobel(suave)
    binaria = umbralizacion_manual(bordes, 35) # <--- AJUSTA ESTE NÚMERO
    
    total = contar_celulas(binaria)
    
    # 2. Mostrar resultados
    print(f"Conteo final: {total} células detectadas.")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(bordes, cmap='gray'); plt.title("Bordes")
    plt.subplot(1, 3, 3); plt.imshow(binaria, cmap='gray'); plt.title(f"Resultado: {total}")
    plt.show()

# EJECUTAR: cambia 'celulas.png' por el nombre de tu archivo
if __name__ == "__main__":
    ejecutar_proyecto('celulas.png')