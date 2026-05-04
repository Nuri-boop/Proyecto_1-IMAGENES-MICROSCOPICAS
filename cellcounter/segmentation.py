import numpy as np

def manual_convolution(image, kernel):
    """Convolución manual 2D."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w))
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    output = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

def detect_edges(image, method='sobel'):
    """Detección de bordes con Sobel manual (Edge detection)."""
    if method == 'sobel':
        Gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        Gx = manual_convolution(image.astype(np.float32), Gx_kernel)
        Gy = manual_convolution(image.astype(np.float32), Gy_kernel)
        
        magnitud = np.sqrt(Gx**2 + Gy**2)
        
        if magnitud.max() > 0:
            magnitud = (magnitud / magnitud.max() * 255).astype(np.uint8)
        
        return magnitud
    else:
        raise ValueError("Solo método 'sobel' implementado")

def threshold_otsu(image):
    """Umbralización de Otsu manual (Thresholding)."""
    hist = np.zeros(256)
    for pixel in image.flatten():
        hist[pixel] += 1
    
    total = image.size
    hist = hist / total
    
    mejor_umbral = 0
    mejor_varianza = 0
    
    for t in range(256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        
        if w0 == 0 or w1 == 0:
            continue
        
        mu0 = 0
        mu1 = 0
        
        if w0 > 0:
            for i in range(t):
                mu0 += i * hist[i]
            mu0 = mu0 / w0
        
        if w1 > 0:
            for i in range(t, 256):
                mu1 += i * hist[i]
            mu1 = mu1 / w1
        
        varianza = w0 * w1 * (mu0 - mu1)**2
        
        if varianza > mejor_varianza:
            mejor_varianza = varianza
            mejor_umbral = t
    
    binary = (image >= mejor_umbral).astype(np.uint8) * 255
    return binary