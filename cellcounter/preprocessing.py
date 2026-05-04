import numpy as np
import cv2

def load_and_gray(path):
    """Carga la imagen y la convierte a escala de grises."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_intensity_transform(image):
    """Estiramiento de contraste manual (Min-Max Scaling)."""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max == img_min:
        return image
    rescaled = (image - img_min) * (255.0 / (img_max - img_min))
    return rescaled.astype(np.uint8)

def frequency_filter(image, radius=30, filter_type='low'):
    """Filtro pasa-bajas o pasa-altas ideal en frecuencia usando FFT."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    
    if filter_type == 'low':
        mask = mask_area.astype(float)
    elif filter_type == 'high':
        mask = (~mask_area).astype(float)
    else:
        raise ValueError("filter_type debe ser 'low' o 'high'")
    
    f_filt = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(f_filt))
    return np.abs(img_back).astype(np.uint8)

def spatial_filter(image, filter_type='median', kernel_size=3):
    """Filtro espacial manual."""
    h, w = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)
    
    if filter_type == 'median':
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(window)
    elif filter_type == 'mean':
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.mean(window)
    elif filter_type == 'gaussian':
        sigma = 1.0
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.sum(window * kernel)
    
    return output.astype(np.uint8)