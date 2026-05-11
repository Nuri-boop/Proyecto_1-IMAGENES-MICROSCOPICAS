# Proyecto_1-IMAGENES-MICROSCOPICAS
Esta librería conta de la detección, segmentación y conteo automatizado de células en imágenes microscópicas.
Automated detection, segmentation and counting of cells in microscopic images.

**Desde VS Code**
----------------------------------------------------------------------------------------------------------------------------
REQUISITOS
  Dependencias
    numpy
    matplotlib
    opencv-python
    scipy
  
1. Descarga o descomprime la carpeta `Proyecto_1-IMAGENES-MICROSCOPICAS-main`.

2. Abre la carpeta y haz clic en la barra de dirección del explorador de archivos.

3. Escribe `cmd` y presiona Enter.  
   Esto abrirá una terminal directamente en la carpeta del proyecto.

4. Instala las dependencias necesarias ejecutando:

  ```bash
    pip install -r requirements.txt 
  ```
  **En caso de contar con las dependencias previamente solo es necesario instalarlo localmente (Paso 5)

  
5. Después instala la librería localmente con:
 ```bash
    pip install .
```

6. Para verificar que la instalación fue correcta, abre Python y ejecuta:
    from cellcounter import preprocessing, segmentation, analysis
   
Si no aparece ningún error, la instalación fue exitosa. 
----------------------------------------------------------------------------------------------------------------------------
