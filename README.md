# Proyecto_1-IMAGENES-MICROSCOPICAS
Esta librería consta de la detección, segmentación y conteo automatizado de células en imágenes microscópicas.
Automated detection, segmentation and counting of cells in microscopic images.

REQUISITOS
  Dependencias
    numpy
    matplotlib
    opencv-python
    scipy
  
**Desde VS Code con la paqueteria distribuida desde GitHub**
----------------------------------------------------------------------------------------------------------------------------
1. Ingresa en la terminal de tu dispositivo:
   ```bash
      pip install https://github.com/Nuri-boop/Proyecto_1-IMAGENES-MICROSCOPICAS/archive/main.zip
    ```
    Sigue las instrucciones para la instalacion de la librería
   
 3. Descarga la carpeta desde el archivo.
 4. Ejecuta el archivo "Proyecto_1.py"
 5. Para verificar que la instalación fue correcta, abre Python y ejecuta:
    from cellcounter import preprocessing, segmentation, analysis
Si no aparece ningún error, la instalación fue exitosa. 
    
----------------------------------------------------------------------------------------------------------------------------


**Desde VS Code desde la carpeta comprimida**
----------------------------------------------------------------------------------------------------------------------------
1. Descarga o descomprime la carpeta `Proyecto_1-IMAGENES-MICROSCOPICAS-main`.

2. Abre la carpeta y haz clic en la barra de dirección del explorador de archivos.

3. Escribe `cmd` y presiona Enter.  
   Esto abrirá una terminal directamente en la carpeta del proyecto.

4. Instala las dependencias necesarias ejecutando:

  ```bash
    pip install -r requirements.txt 
  ```
  **En caso de contar previamente con las dependencias solo es necesario instalarlo localmente (Paso 5)

  
5. Después instala la librería localmente con:
 ```bash
    pip install .
```

6. Para verificar que la instalación fue correcta, abre Python y ejecuta:
    from cellcounter import preprocessing, segmentation, analysis
   
Si no aparece ningún error, la instalación fue exitosa. 

----------------------------------------------------------------------------------------------------------------------------

** Desde Spyder usando Anaconda **
----------------------------------------------------------------------------------------------------------------------------

1. Abre Anaconda Prompt.

2. Navega hasta la carpeta del proyecto usando el comando `cd`. Por ejemplo:

  ```bash
    cd C:\Users\TuUsuario\Downloads\Proyecto_1-IMAGENES-MICROSCOPICAS-main
  ```

3. Instala las dependencias necesarias:
  ```bash
    pip install -r requirements.txt
  ```
 **En caso de contar previamente con las dependencias solo es necesario instalarlo localmente (Paso 4)

4. Instala la librería localmente:
  ```bash 
    pip install .
  ```
5. Una vez finalizada la instalación, abre Spyder.
6. Verifica que la librería funciona ejecutando:
    from cellcounter import preprocessing, segmentation, analysis

Si no aparece ningún error, la instalación fue exitosa y la librería ya puede utilizarse desde Spyder.

----------------------------------------------------------------------------------------------------------------------------
